from decimal import Decimal
from typing import Dict, List, Set
from pydantic import Field, validator, ConfigDict
import time
import csv
import os
import math
from enum import Enum

from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.core.data_type.common import PriceType, TradeType
from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.executors.data_types import ConnectorPair
from hummingbot.client.config.config_data_types import ClientFieldData
from pydantic import BaseModel
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.models.base import RunnableStatus


PAIR_ONE_ENTRY_EXECUTOR_ID = "PAIR_ONE_ENTRY"
PAIR_TWO_ENTRY_EXECUTOR_ID = "PAIR_TWO_ENTRY"



class PositionStatus(Enum):
    NO_POSITION = "no_position"
    ENTRY_IN_PROGRESS = "entry_in_progress"
    IN_POSITION = "in_position"
    SL_IN_PROGRESS = "sl_in_progress"
    TP_IN_PROGRESS = "tp_in_progress"
    TIMEOUT_EXIT_IN_PROGRESS = "timeout_exit_in_progress"

class ExecutorState(BaseModel):
    id: str
    timestamp: float
    type: str
    close_timestamp: float | None
    close_type: CloseType | None
    status: RunnableStatus
    config: PositionExecutorConfig
    net_pnl_pct: Decimal
    net_pnl_quote: Decimal
    cum_fees_quote: Decimal
    filled_amount_quote: Decimal
    is_active: bool
    is_trading: bool
    custom_info: dict
    controller_id: str | None = None

    @property
    def level_id(self) -> str:
        return self.custom_info.get("level_id", "unknown_level_id")

    @property
    def is_active_and_not_filled(self) -> bool:
        return self.is_active and not self.is_trading

    @property
    def is_full_filled(self) -> bool:
        return self.is_active and self.is_trading

    @property
    def average_entry_price(self) -> float:
        return float(self.custom_info.get("current_position_average_price", 0.0))

    @property
    def amount(self) -> float:
        return float(self.filled_amount_quote)

    @property
    def qty(self) -> float:
        if self.average_entry_price == 0:
            return 0.0
        return float(self.amount / self.average_entry_price)

    @property
    def config_qty(self) -> float:
        return float(self.config.amount)

    @property
    def config_entry_price(self) -> float:
        return float(self.config.entry_price)

    @property
    def is_terminated(self) -> bool:
        return self.status in [
            RunnableStatus.TERMINATED, 
            # RunnableStatus.SHUTTING_DOWN
        ]


class DeltaArbitrageControllerV1Config(ControllerConfigBase):
    model_config = ConfigDict(extra="allow")
    controller_name: str = "delta_arbitrage_controller_v1"
    candles_config: List[CandlesConfig] = Field(
        default="bybit_perpetual.BTC-USDT.1m.1000:bybit_perpetual.ETH-USDT.1m.1000:",
        client_data=ClientFieldData(
            is_updatable=True,
            prompt_on_new=True,
            prompt=lambda mi: (
                "Enter candle configs in format 'exchange1.tp1.interval1.max_records':"
                "'exchange2.tp2.interval2.max_records'"
            ),
        ),
    )
    connector_one: str = Field(
        default="bybit_perpetual",
        client_data=ClientFieldData(prompt=lambda e: "Enter the connector ONE: ", prompt_on_new=True),
    )
    trading_pair_one: str = Field(
        default="BTC-USDT",
        client_data=ClientFieldData(prompt=lambda e: "Enter the trading pair ONE: ", prompt_on_new=True),
    )
    connector_two: str = Field(
        default="bybit_perpetual",
        client_data=ClientFieldData(prompt=lambda e: "Enter the connector TWO: ", prompt_on_new=True),
    )
    trading_pair_two: str = Field(
        default="ETH-USDT",
        client_data=ClientFieldData(prompt=lambda e: "Enter the trading pair TWO: ", prompt_on_new=True),
    )
    delta_calculation_interval: str = Field(
        default="1m",
        client_data=ClientFieldData(prompt=lambda e: "Enter the regression interval (1m): ", prompt_on_new=True),
    )
    reference_price_window: int = Field(
        default=100,
        client_data=ClientFieldData(prompt=lambda e: "Enter the reference price window size: ", prompt_on_new=True),
    )
    position_max_duration_sec: int = Field(
        default=60000,  # 1000 minutes
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the period to close position by timeout in seconds (60000): ", prompt_on_new=True
        ),
    )
    # Ratio thresholds
    open_position_lower_threshold: float = Field(
        default=8.0,
        client_data=ClientFieldData(prompt=lambda e: "Enter the open lower threshold: ", prompt_on_new=True),
    )  # Sell BTC/Buy ETH when 10 >= BTC/ETH >= 8
    open_position_upper_threshold: float = Field(
        default=10.0,
        client_data=ClientFieldData(prompt=lambda e: "Enter the open upper threshold: ", prompt_on_new=True),
    )
    take_profit_threshold: float = Field(
        default=5.0,
        client_data=ClientFieldData(prompt=lambda e: "Enter the take profit threshold: ", prompt_on_new=True),
    )  # Exit BTC short/ETH long when profit reaches >= 0.005
    stop_loss_threshold: float = Field(
        default=3.0, client_data=ClientFieldData(prompt=lambda e: "Enter the stop loss threshold: ", prompt_on_new=True)
    )  # Exit BTC short/ETH long when loss reaches <= -0.002

    fee_decimal: float = Field(
        default=0.001, client_data=ClientFieldData(prompt=lambda e: "Enter the fee decimal: ", prompt_on_new=True)
    )
    total_amount_quote: float = Field(
        default=100.0,
        client_data=ClientFieldData(prompt=lambda e: "Enter the total amount in quote currency: ", prompt_on_new=True)
    )
    manual_kill_switch: bool = Field(
        default=False,
        client_data=ClientFieldData(prompt=lambda e: "Enable manual kill switch? (True/False): ", prompt_on_new=True)
    )

    @validator("candles_config", pre=True)
    def parse_candles_config(cls, v) -> List[CandlesConfig]:
        if isinstance(v, str):
            return cls.parse_candles_config_str(v)
        elif isinstance(v, list):
            return v
        raise ValueError("Invalid type for candles_config. Expected str or List[CandlesConfig]")

    @validator("total_amount_quote", pre=True)
    def parse_total_amount_quote(cls, v) -> float:
        return float(v)

    @validator("fee_decimal", pre=True)
    def parse_fee_decimal(cls, v) -> float:
        return float(v)

    @staticmethod
    def parse_candles_config_str(v: str) -> List[CandlesConfig]:
        configs = []
        if v.strip():
            entries = v.split(":")
            for entry in entries:
                parts = entry.split(".")
                if len(parts) != 4:
                    raise ValueError(
                        f"Invalid candles config format in segment '{entry}'. "
                        "Expected format: 'exchange.tradingpair.interval.maxrecords'"
                    )
                connector, trading_pair, interval, max_records_str = parts
                try:
                    max_records = int(max_records_str)
                except ValueError:
                    raise ValueError(
                        f"Invalid max_records value '{max_records_str}' in segment '{entry}'. "
                        "max_records should be an integer."
                    )
                config = CandlesConfig(
                    connector=connector, trading_pair=trading_pair, interval=interval, max_records=max_records
                )
                configs.append(config)
        return configs

    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        if self.connector_one not in markets:
            markets[self.connector_one] = set()
        markets[self.connector_one].add(self.trading_pair_one)
        if self.connector_two not in markets:
            markets[self.connector_two] = set()
        markets[self.connector_two].add(self.trading_pair_two)
        return markets


class DeltaArbitrageControllerV1(ControllerBase):

    @property
    def config_total_amount_quote(self) -> float:
        return float(self.config.total_amount_quote)


    @property
    def executors_states(self) -> dict[str, ExecutorState]:
        executors_states = {ex.custom_info["level_id"]: ExecutorState(**ex.to_dict()) for ex in self.executors_info}
        return executors_states

    def __init__(self, config: DeltaArbitrageControllerV1Config, *args, **kwargs):
        self.logger().warning(f"Initializing Delta Arbitrage Controller with config: {config}")
        super().__init__(config, *args, **kwargs)
        self.config = config
        # self.config.position_max_duration_sec = 1000 * 60 # 1000 minutes
        self.position_status = PositionStatus.NO_POSITION
        # self._has_active_position = False
        # self._is_position_entry_in_progress = False
        # self._is_take_profit_in_progress = False
        # self._is_stop_loss_in_progress = False
        self._is_first_tick = True
        self.pair_one_current_price: float = 0.0
        self.pair_two_current_price: float = 0.0
        self.pair_one_current_qty: float = 0.0
        self.pair_two_current_qty: float = 0.0
        self._current_ratio: float = 0.0
        self.reference_price_one: float = 0.0
        self.reference_price_two: float = 0.0
        self._position_open_price_one: float = 0.0
        self._position_open_price_two: float = 0.0
        self._current_delta: float = 0.0
        self._reference_delta: float = 0.0
        self.fee_decimal: float = float(config.fee_decimal)
        self._reference_prices_updated_at: int = 0
        self._reference_delta_updated_at: int = 0
        self.current_amount_quote: float = float(config.total_amount_quote)
        self._current_position_opened_at: int = 0
        self._current_position_timeout_at: int = 0
        self.cumulative_delta: float = 1.0
        self._total_positions_opened_count: int = 0
        self._total_positions_closed_count: int = 0
        self._positions_closed_by_timeout_count: int = 0
        self._positions_closed_by_stop_loss_count: int = 0
        self._positions_closed_by_take_profit_count: int = 0
        self.executors_actions: list[ExecutorAction] = []
        self.market_data_provider.initialize_rate_sources(
            [
                ConnectorPair(connector_name=config.connector_one, trading_pair=config.trading_pair_one),
                ConnectorPair(connector_name=config.connector_two, trading_pair=config.trading_pair_two),
            ]
        )
        self._pair_one_active_executor_id: str | None = None
        self._pair_two_active_executor_id: str | None = None

    def start(self):
        """
        Действия, выпоенные при старте контроллера, на первом тике
        """
        self.logger().info("Starting Delta Arbitrage Controller...")
        super().start()
        self.on_new_cycle_start()

    def on_new_cycle_start(self):
        """
        Цикл завершается по времени или после первой закрытой сделки
        """
        self.logger().info("Starting new cycle...")
        for executor in self.executors_info:
            self.logger().info(f"Executor {executor}")
        current_time = int(time.time())
        self._current_position_opened_at = 0
        self._current_position_timeout_at = 0
        self.position_status = PositionStatus.NO_POSITION
        self._pair_one_active_executor_id: str | None = None
        self._pair_two_active_executor_id: str | None = None
        self.update_reference_data(current_time)
        return None

    def update_reference_data(self, current_time: int):
        """
        Update historical prices for delta calculation
        """
        self._reference_delta = self.calculate_reference_relative_delta()
        self._reference_delta_updated_at = current_time

        REFERENCE_PRICES_UPDATE_INTERVAL = 60
        time_since_reference_updated = current_time - self._reference_prices_updated_at
        if time_since_reference_updated <= REFERENCE_PRICES_UPDATE_INTERVAL:
            return None

        ohlcv_df_one = self.market_data_provider.get_candles_df(
            connector_name=self.config.connector_one,
            trading_pair=self.config.trading_pair_one,
            interval=self.config.delta_calculation_interval,
            max_records=self.config.reference_price_window,
        )

        ohlcv_df_two = self.market_data_provider.get_candles_df(
            connector_name=self.config.connector_two,
            trading_pair=self.config.trading_pair_two,
            interval=self.config.delta_calculation_interval,
            max_records=self.config.reference_price_window,
        )
        self.reference_price_one = ohlcv_df_one["close"].values[0]
        self.reference_price_two = ohlcv_df_two["close"].values[0]
        self._reference_prices_updated_at = current_time
        return None

    def calculate_reference_relative_delta(self) -> float:
        if (
            self.reference_price_one == 0
            or self.reference_price_two == 0
            or self.pair_one_current_price == 0
            or self.pair_two_current_price == 0
        ):
            return 0.0

        return self.calculate_relative_delta_short(
            init_price_one=self.reference_price_one,
            last_price_one=self.pair_one_current_price,
            init_price_two=self.reference_price_two,
            last_price_two=self.pair_two_current_price,
            fee_decimal=self.fee_decimal,
        )

    def calculate_current_relative_delta(self) -> float:
        # self.logger().info(f"Current relative delta: {self._current_delta}")
        # self.logger().info(f"Position open price one: {self._position_open_price_one}")
        # self.logger().info(f"Position open price two: {self._position_open_price_two}")
        # self.logger().info(f"Pair one current price: {self.pair_one_current_price}")
        # self.logger().info(f"Pair two current price: {self.pair_two_current_price}\n\n")
        return self.calculate_relative_delta_short(
            init_price_one=self._position_open_price_one,
            last_price_one=self.pair_one_current_price,
            init_price_two=self._position_open_price_two,
            last_price_two=self.pair_two_current_price,
            fee_decimal=self.fee_decimal,
        )

    def calculate_relative_delta_short(
        self,
        init_price_one: float,
        last_price_one: float,
        init_price_two: float,
        last_price_two: float,
        fee_decimal: float,
    ) -> float:

        if init_price_one == 0 or init_price_two == 0:
            return 0.0

        pair_one_ratio = last_price_one / init_price_one
        pair_two_ratio = last_price_two / init_price_two

        delta = pair_two_ratio - pair_one_ratio
        fee = fee_decimal * (pair_one_ratio + pair_two_ratio + 2)
        return delta - fee


    def _is_position_open_conditions(self) -> bool:
        return (
            self.config.open_position_lower_threshold
            <= self._reference_delta
            <= self.config.open_position_upper_threshold
        )


    def _check_and_handle_stop_loss(self) -> None:
        if self._current_delta <= self.config.stop_loss_threshold:
            self._start_stop_loss_exit_process()
        return None


    def _start_stop_loss_exit_process(self):
        self.logger().info("Starting stop loss exit process...")
        self._start_close_position_process()
        self.position_status = PositionStatus.SL_IN_PROGRESS
        return None


    def _check_and_handle_take_profit(self) -> None:
        if self._current_delta >= self.config.take_profit_threshold:
            self._start_take_profit_exit_process()
        return None


    def _start_take_profit_exit_process(self):
        self.logger().info("Starting take profit exit process...")
        self._start_close_position_process()
        self.position_status = PositionStatus.TP_IN_PROGRESS
        return None


    def _start_position_entry_process(self):
        # create sell executor for pair One
        self.logger().info("Starting position entry process...")
        self.executors_actions.append(
                CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=self.get_pair_one_market_entry_executor_config(),
                ),
            )
        # create buy executor for pair Two
        self.executors_actions.append(
                CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=self.get_pair_two_market_entry_executor_config(),
                ),
            )
        # self.open_position()
        # self._is_position_entry_in_progress = True
        self.position_status = PositionStatus.ENTRY_IN_PROGRESS
        return None

    def _check_and_handle_position_entry_process(self):
        # check entry executors statuses
        self.logger().info("Checking entry executors statuses...")
        self._pair_one_active_executor_id = None
        self._pair_two_active_executor_id = None

        for executor in self.executors_info:
            self.logger().info(f"EXECUTOR: {executor}")
            if executor.status == RunnableStatus.RUNNING:
                if executor.custom_info["level_id"] == PAIR_ONE_ENTRY_EXECUTOR_ID:
                    self._pair_one_active_executor_id = executor.id
                    self.logger().info(f"Pair One active executor: {executor.id}")
                elif executor.custom_info["level_id"] == PAIR_TWO_ENTRY_EXECUTOR_ID:
                    self._pair_two_active_executor_id = executor.id
                    self.logger().info(f"Pair Two active executor: {executor.id}")

                if self._pair_one_active_executor_id and self._pair_two_active_executor_id:
                    self.logger().info("POSITIONS OPENED")
                    self._total_positions_opened_count += 1
                    self.position_status = PositionStatus.IN_POSITION
                    self._position_open_price_one = self.pair_one_current_price
                    self._position_open_price_two = self.pair_two_current_price
                    self._current_position_opened_at = int(time.time())
                    self._current_position_timeout_at = self._current_position_opened_at + self.config.position_max_duration_sec
                    break

        return None


    def _check_and_handle_position_timeout(self, current_time: int) -> None:
        if current_time > self._current_position_timeout_at:
            if self._is_position_open_conditions():
                self._current_position_timeout_at += self.config.position_max_duration_sec
                self.logger().info("Position timeout has been prolonged")
                return None
            self._start_position_timeout_exit_process()
        return None


    def _start_position_timeout_exit_process(self):
        self._start_close_position_process()
        self.position_status = PositionStatus.TIMEOUT_EXIT_IN_PROGRESS
        

    def _check_and_handle_timeout_exit_process(self) -> None:
        if self._is_position_closed():
            self._positions_closed_by_timeout_count += 1
            self._on_position_close()
        return None
       

    def _is_position_closed(self) -> bool:
        # TODO: optimize algo 
        is_executor_one_terminated = False
        is_executor_two_terminated = False

        self.logger().info("IS_POSITION_CLOSED?")
        self.logger().info(f"Pair One active executor: {self._pair_one_active_executor_id}")
        self.logger().info(f"Pair Two active executor: {self._pair_two_active_executor_id}")
        
        for executor in self.executors_info:
            self.logger().info(f"Executor: {executor}")
            if executor.id == self._pair_one_active_executor_id:
                is_executor_one_terminated = executor.status == RunnableStatus.TERMINATED
            elif executor.id == self._pair_two_active_executor_id:
                is_executor_two_terminated = executor.status == RunnableStatus.TERMINATED

            if is_executor_one_terminated and is_executor_two_terminated:
               return True
        self.logger().info(f"is_executor_one_terminated: {is_executor_one_terminated}")
        self.logger().info(f"is_executor_two_terminated: {is_executor_two_terminated}")
        return False


    def _check_and_handle_if_position_opened(self) -> bool:
        executor_one = None
        executor_two = None
        for executor in self.executors_info:
            if executor.status == RunnableStatus.RUNNING:
                if executor.custom_info["level_id"] == PAIR_ONE_ENTRY_EXECUTOR_ID:
                    self._pair_one_active_executor_id = executor.id
                    executor_one = executor
                elif executor.custom_info["level_id"] == PAIR_TWO_ENTRY_EXECUTOR_ID:
                    self._pair_two_active_executor_id = executor.id
                    executor_two = executor
            if executor_one and executor_two:
                self._position_open_price_one = float(executor_one.custom_info["current_position_average_price"])
                self._position_open_price_two = float(executor_two.custom_info["current_position_average_price"])
                if executor_one.custom_info["open_order_last_update"]:
                    self._current_position_opened_at = int(executor_one.custom_info["open_order_last_update"])   
                else:
                    self._current_position_opened_at = int(executor.timestamp)
                self._current_position_timeout_at = self._current_position_opened_at + self.config.position_max_duration_sec
                return True
        return False


    def _on_position_close(self):
        self.logger().info("POSITION CLOSED...")
        self.position_status = PositionStatus.NO_POSITION

        executor_1 = next((e for e in self.executors_info if e.custom_info["level_id"] == self._pair_one_active_executor_id), None)
        executor_2 = next((e for e in self.executors_info if e.custom_info["level_id"] == self._pair_two_active_executor_id), None)
        
        if not executor_1 or not executor_2:
            raise ValueError(f"Executor {self._pair_one_active_executor_id} or {self._pair_two_active_executor_id} not found")

        # calculate pnl 
        net_pnl_quote = executor_1.net_pnl_quote + executor_2.net_pnl_quote
        self.cumulative_delta = self.cumulative_delta * (1 + self._current_delta)


        self.logger().info("POSITION CLOSED...")
        self.logger().info(f"Executor 1: {executor_1}")
        self.logger().info(f"Executor 2: {executor_2}")
        self.logger().info(f"Net PNL Quote: {net_pnl_quote}")
        self.current_amount_quote = self.current_amount_quote + net_pnl_quote
        self.logger().info(f"Current Amount Quote: {self.current_amount_quote}")

        self._pair_one_active_executor_id = None
        self._pair_two_active_executor_id = None
        self.on_new_cycle_start()
        return None


    def _check_and_handle_stop_loss_exit_process(self) -> None:
        if self._is_position_closed():
            self._positions_closed_by_stop_loss_count += 1
            self._on_position_close()
        return None


    def _check_and_handle_take_profit_exit_process(self) -> None:
        if self._is_position_closed():
            self._positions_closed_by_take_profit_count += 1
            self._on_position_close()
        return None

    # def open_position(self):
    #     self._position_open_price_one = self.pair_one_current_price
    #     self._position_open_price_two = self.pair_two_current_price
    #     self._current_position_opened_at = int(time.time())
    #     self._total_positions_opened_count += 1
    #     # self._has_active_position = True
    #     self.position_status = PositionStatus.IN_POSITION

    def _start_close_position_process(self):
        self.executors_actions.append(
            StopExecutorAction(
                controller_id=self.config.id,
                executor_id=self._pair_one_active_executor_id,
            )
        )
        self.executors_actions.append(
            StopExecutorAction(
                controller_id=self.config.id,
                executor_id=self._pair_two_active_executor_id,
            )
        )

        # self.cumulative_delta = self.cumulative_delta * (1 + self._current_delta)
        # self.current_amount_quote = self.current_amount_quote + self._current_delta * self.config_total_amount_quote
        # self._position_open_price_one = 0.0
        # self._position_open_price_two = 0.0
        # self._total_positions_closed_count += 1
        # self._current_position_opened_at = 0
        # # self._has_active_position = False
        # self.position_status = PositionStatus.NO_POSITION


    def quantize_qty(self, qty: float) -> float:
        qty_quantized = self.market_data_provider.quantize_order_amount(
            self.config.connector_name, self.config.trading_pair, Decimal(str(qty))
        )
        qty_quantized_float = float(qty_quantized)
        if math.isnan(qty_quantized_float):
            self.logger().warning("Quantized qty is NaN. Using original qty: %s", qty)
            qty_quantized_float = qty
        return qty_quantized_float


    def get_pair_one_market_entry_executor_config(self) -> PositionExecutorConfig:
        amount = self.current_amount_quote / 2
        price = self.pair_one_current_price
        qty = amount / price
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_one,
            trading_pair=self.config.trading_pair_one,
            side=TradeType.SELL,
            amount=qty,
            level_id=PAIR_ONE_ENTRY_EXECUTOR_ID,
            position_mode="ONEWAY",
            triple_barrier_config=TripleBarrierConfig(
                open_order_type=OrderType.MARKET,
            ),
        )

    def get_pair_two_market_entry_executor_config(self) -> PositionExecutorConfig:
        amount = self.current_amount_quote / 2
        price = self.pair_two_current_price
        qty = amount / price
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_two,
            trading_pair=self.config.trading_pair_two,
            side=TradeType.BUY,
            amount=qty,
            level_id=PAIR_TWO_ENTRY_EXECUTOR_ID,
            position_mode="ONEWAY",
            triple_barrier_config=TripleBarrierConfig(
                open_order_type=OrderType.MARKET,
            ),
        )

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """Determine what positions to open based on the BTC/ETH ratio"""
        executor_actions = self.executors_actions
        self.executors_actions = []
        return executor_actions


    async def update_processed_data(self):
        """Update any processed data needed by the controller"""
        if self._is_first_tick:
            self._is_first_tick = False
            self.logger().info("First tick detected, starting controller...")
            if self._check_and_handle_if_position_opened():
                self.position_status = PositionStatus.IN_POSITION
                self.logger().info("POSITION ALREADY OPENED in first tick")
                return None
                
            self.on_new_cycle_start()
            return None

        tick_current_time = int(time.time())
        # Получить свежие цены current_price_one и current_price_two
        self.pair_one_current_price = float(
            self.market_data_provider.get_price_by_type(
                self.config.connector_one, self.config.trading_pair_one, PriceType.LastTrade
            )
        )
        self.pair_two_current_price = float(
            self.market_data_provider.get_price_by_type(
                self.config.connector_two, self.config.trading_pair_two, PriceType.LastTrade
            )
        )

        self.update_reference_data(tick_current_time)
        self.logger().info(f"STATUS: {self.position_status}")

        if self.position_status == PositionStatus.ENTRY_IN_PROGRESS:
            self._check_and_handle_position_entry_process()
            
        elif self.position_status == PositionStatus.TP_IN_PROGRESS:
            self._check_and_handle_take_profit_exit_process()

        elif self.position_status == PositionStatus.SL_IN_PROGRESS:
            self._check_and_handle_stop_loss_exit_process()
            
        elif self.position_status == PositionStatus.TIMEOUT_EXIT_IN_PROGRESS:
            self._check_and_handle_timeout_exit_process()

        elif self.position_status == PositionStatus.IN_POSITION:
            # self.logger().info("IN_POSITION - Checking position conditions...")
            self._current_delta = self.calculate_current_relative_delta()
            self._check_and_handle_stop_loss()
            if self.position_status == PositionStatus.IN_POSITION:
                self._check_and_handle_take_profit()
            if self.position_status == PositionStatus.IN_POSITION:
                self._check_and_handle_position_timeout(tick_current_time)

        elif self.position_status == PositionStatus.NO_POSITION:
            # self.logger().info("NO POSITION - Checking position open conditions...")
            if self._check_and_handle_if_position_opened():
                self.position_status = PositionStatus.IN_POSITION
                self._total_positions_opened_count += 1
                self.logger().info("POSITION ALREADY OPENED")
            else: 
                self._check_and_handle_position_open_conditions()

        save_controller_state_to_file(self, tick_current_time)
        return None


    def _check_and_handle_position_open_conditions(self):
        self.logger().info(f"Checking position open conditions: {self._is_position_open_conditions()}")

        if self._is_position_open_conditions():
            self._start_position_entry_process()


    def to_format_status(self) -> List[str]:
        """Format the current status for display"""
        status = [
            f"{self.config.trading_pair_one} price: {self.pair_one_current_price}",
            f"{self.config.trading_pair_two} price: {self.pair_two_current_price}",
            f"Current Delta: {self._current_delta}",
            f"Reference Delta: {self._reference_delta}",
            f"Position open lower threshold: {self.config.open_position_lower_threshold}",
            f"Position open upper threshold: {self.config.open_position_upper_threshold}",
            f"Position take profit threshold: {self.config.take_profit_threshold}",
            f"Position stop loss threshold: {self.config.stop_loss_threshold}",
            f"Position max duration: {self.config.position_max_duration_sec} seconds",
            f"Position status: {self.position_status.value.upper()}",
            f"Active executor ONE: {self._pair_one_active_executor_id}",
            f"Active executor TWO: {self._pair_two_active_executor_id}",
            f"Total positions opened count: {self._total_positions_opened_count}",
            f"Total positions closed count: {self._total_positions_closed_count}",
            f"Positions closed by timeout count: {self._positions_closed_by_timeout_count}",
            f"Positions closed by take profit count: {self._positions_closed_by_take_profit_count}",
            f"Positions closed by stop loss count: {self._positions_closed_by_stop_loss_count}",
            f"Current position opened at: {self._current_position_opened_at}",
            f"Reference delta updated at: {self._reference_delta_updated_at}",
            f"Reference prices updated at: {self._reference_prices_updated_at}",
            f"Position open price ONE: {self._position_open_price_one}",
            f"Position open price TWO: {self._position_open_price_two}",
            f"Time until timeout: {self._current_position_timeout_at - int(time.time())}"
            f"Cumulative delta: {self.cumulative_delta}",
            f"Current amount quote: {self.current_amount_quote}",
        ]

        return status


def save_controller_state_to_file(controller: DeltaArbitrageControllerV1, timestamp: int):
    filename = (
        f"logs/controller_state_history_{controller.config.controller_name}_{controller.config.connector_one}.csv"
    )
    headers = [
        "timestamp",
        "pair_one",
        "pair_two",
        "pair_one_price",
        "pair_two_price",
        "reference_price_one",
        "reference_price_two",
        "reference_prices_updated_at",
        "reference_delta_updated_at",
        "reference_delta",
        "current_delta",
        "position_status",
        "position_open_lower_threshold",
        "position_open_upper_threshold",
        "position_take_profit_threshold",
        "position_stop_loss_threshold",
        "current_position_opened_at",
        "total_positions_opened_count",
        "total_positions_closed_count",
        "positions_closed_by_timeout_count",
        "positions_closed_by_take_profit_count",
        "positions_closed_by_stop_loss_count",
        "cumulative_delta",
        "current_amount_quote",
    ]

    row = {
        "timestamp": timestamp,
        "pair_one": controller.config.trading_pair_one,
        "pair_two": controller.config.trading_pair_two,
        "pair_one_price": controller.pair_one_current_price,
        "pair_two_price": controller.pair_two_current_price,
        "reference_price_one": controller.reference_price_one,
        "reference_price_two": controller.reference_price_two,
        "reference_prices_updated_at": controller._reference_prices_updated_at,
        "reference_delta_updated_at": controller._reference_delta_updated_at,
        "reference_delta": controller._reference_delta,
        "current_delta": controller._current_delta,
        "position_status": controller.position_status.value,
        "position_open_lower_threshold": controller.config.open_position_lower_threshold,
        "position_open_upper_threshold": controller.config.open_position_upper_threshold,
        "position_take_profit_threshold": controller.config.take_profit_threshold,
        "position_stop_loss_threshold": controller.config.stop_loss_threshold,
        "current_position_opened_at": controller._current_position_opened_at,
        "total_positions_opened_count": controller._total_positions_opened_count,
        "total_positions_closed_count": controller._total_positions_closed_count,
        "positions_closed_by_timeout_count": controller._positions_closed_by_timeout_count,
        "positions_closed_by_take_profit_count": controller._positions_closed_by_take_profit_count,
        "positions_closed_by_stop_loss_count": controller._positions_closed_by_stop_loss_count,
        "cumulative_delta": controller.cumulative_delta,
        "current_amount_quote": controller.current_amount_quote,
    }

    if not os.path.exists(filename):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(row.values())
    else:
        with open(filename, "a") as f:
            writer = csv.writer(f)
            writer.writerow(row.values())
