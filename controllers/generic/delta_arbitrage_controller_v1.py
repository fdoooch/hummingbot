from decimal import Decimal
from typing import Dict, List, Set
from pydantic import Field, validator, ConfigDict
import time
import csv
import os

from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.core.data_type.common import PriceType
from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction
from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType
from hummingbot.strategy_v2.executors.data_types import ConnectorPair
from hummingbot.client.config.config_data_types import ClientFieldData


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

    def __init__(self, config: DeltaArbitrageControllerV1Config, *args, **kwargs):
        self.logger().warning(f"Initializing Delta Arbitrage Controller with config: {config}")
        super().__init__(config, *args, **kwargs)
        self.config = config
        # self.config.position_max_duration_sec = 1000 * 60 # 1000 minutes
        self._has_active_position = False
        self._is_first_tick = True
        self.pair_one_current_price: float = 0.0
        self.pair_two_current_price: float = 0.0
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
        self.cumulative_delta: float = 1.0
        self._total_positions_opened_count: int = 0
        self._total_positions_closed_count: int = 0
        self._positions_closed_by_timeout_count: int = 0
        self._positions_closed_by_stop_loss_count: int = 0
        self._positions_closed_by_take_profit_count: int = 0
        self.market_data_provider.initialize_rate_sources(
            [
                ConnectorPair(connector_name=config.connector_one, trading_pair=config.trading_pair_one),
                ConnectorPair(connector_name=config.connector_two, trading_pair=config.trading_pair_two),
            ]
        )

    def start(self):
        """
        Действия, выпоенные при старте контроллера, на первом тике
        """
        self.logger().info("Starting Ratio Trading Controller...")
        super().start()
        self.on_new_cycle_start()

    def on_new_cycle_start(self):
        """
        Цикл завершается по времени или после первой закрытой сделки
        """
        current_time = int(time.time())
        self._current_position_opened_at = 0
        self._has_active_position = False
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

    def is_position_open_conditions(self) -> bool:
        return (
            self.config.open_position_lower_threshold
            <= self._reference_delta
            <= self.config.open_position_upper_threshold
        )

    def check_and_handle_stop_loss(self) -> None:
        if not self._has_active_position:
            return None
        if self._current_delta <= self.config.stop_loss_threshold:
            self.on_stop_loss()
        return None

    def check_and_handle_take_profit(self) -> None:
        if not self._has_active_position:
            return None
        if self._current_delta >= self.config.take_profit_threshold:
            self.on_take_profit()
        return None

    def on_take_profit(self):
        self.close_position()
        self._positions_closed_by_take_profit_count += 1
        self.on_new_cycle_start()

    def check_and_handle_position_timeout(self, current_time: int) -> None:
        if not self._has_active_position:
            return None
        position_duration = current_time - self._current_position_opened_at
        if position_duration > self.config.position_max_duration_sec:
            self.on_position_timeout()
        return None

    def on_position_timeout(self):
        self.close_position()
        self._positions_closed_by_timeout_count += 1
        self.on_new_cycle_start()

    def on_stop_loss(self):
        self.close_position()
        self._positions_closed_by_stop_loss_count += 1
        self.on_new_cycle_start()

    def open_position(self):
        self._position_open_price_one = self.pair_one_current_price
        self._position_open_price_two = self.pair_two_current_price
        self._current_position_opened_at = int(time.time())
        self._total_positions_opened_count += 1
        self._has_active_position = True

    def close_position(self):
        self.cumulative_delta = self.cumulative_delta * (1 + self._current_delta)
        self.current_amount_quote = self.current_amount_quote + self._current_delta * self.config_total_amount_quote
        self._position_open_price_one = 0.0
        self._position_open_price_two = 0.0
        self._total_positions_closed_count += 1
        self._current_position_opened_at = 0
        self._has_active_position = False

    def create_position_config(
        self,
        trading_pair: str,
        side: TradeType,
        amount_quote: Decimal,
        position_action: PositionAction = PositionAction.OPEN,
    ) -> PositionExecutorConfig:
        """Create a position executor configuration"""
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            trading_pair=trading_pair,
            side=side,
            amount=amount_quote,
            order_type=OrderType.MARKET,
            position_action=position_action,
            take_profit=self.config.take_profit_pct if position_action == PositionAction.OPEN else None,
            stop_loss=self.config.stop_loss_pct if position_action == PositionAction.OPEN else None,
            time_limit_seconds=self.config.time_limit_seconds if position_action == PositionAction.OPEN else None,
            connector_name=self.config.first_pair.connector_name,  # Both pairs use same connector (Binance)
        )

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """Determine what positions to open based on the BTC/ETH ratio"""
        executor_actions = []

        return executor_actions

        # Skip if prices are not valid
        if self._first_pair_last_price <= 0 or self._second_pair_last_price <= 0:
            return executor_actions

        # Check for exit conditions first
        if self._has_active_position and self._current_ratio >= self.config.exit_long_ratio:
            # Close BTC long and ETH short positions
            btc_config = self.create_position_config(
                self.config.first_pair.trading_pair,
                TradeType.SELL,  # Close long with sell
                self.config.amount_quote_btc,
                position_action=PositionAction.CLOSE,
            )
            eth_config = self.create_position_config(
                self.config.second_pair.trading_pair,
                TradeType.BUY,  # Close short with buy
                self.config.amount_quote_eth,
                position_action=PositionAction.CLOSE,
            )
            executor_actions.extend(
                [
                    CreateExecutorAction(executor_config=btc_config, controller_id=self.config.id),
                    CreateExecutorAction(executor_config=eth_config, controller_id=self.config.id),
                ]
            )
            self._has_active_position = False

        elif self._active_first_pair_short and self._current_ratio <= self.config.exit_short_ratio:
            # Close BTC short and ETH long positions
            btc_config = self.create_position_config(
                self.config.first_pair.trading_pair,
                TradeType.BUY,  # Close short with buy
                self.config.amount_quote_btc,
                position_action=PositionAction.CLOSE,
            )
            eth_config = self.create_position_config(
                self.config.second_pair.trading_pair,
                TradeType.SELL,  # Close long with sell
                self.config.amount_quote_eth,
                position_action=PositionAction.CLOSE,
            )
            executor_actions.extend(
                [
                    CreateExecutorAction(executor_config=btc_config, controller_id=self.config.id),
                    CreateExecutorAction(executor_config=eth_config, controller_id=self.config.id),
                ]
            )
            self._active_first_pair_short = False

        # Check for entry conditions
        elif self._current_ratio <= self.config.lower_ratio and not self._has_active_position:
            # Buy BTC and Sell ETH when ratio <= 5
            btc_config = self.create_position_config(
                self.config.first_pair.trading_pair, TradeType.BUY, self.config.amount_quote_btc
            )
            eth_config = self.create_position_config(
                self.config.second_pair.trading_pair, TradeType.SELL, self.config.amount_quote_eth
            )
            executor_actions.extend(
                [
                    CreateExecutorAction(executor_config=btc_config, controller_id=self.config.id),
                    CreateExecutorAction(executor_config=eth_config, controller_id=self.config.id),
                ]
            )
            self._has_active_position = True
            self._active_first_pair_short = False

        elif self._current_ratio >= self.config.upper_ratio and not self._active_first_pair_short:
            # Sell BTC and Buy ETH when ratio >= 6
            btc_config = self.create_position_config(
                self.config.first_pair.trading_pair, TradeType.SELL, self.config.amount_quote_btc
            )
            eth_config = self.create_position_config(
                self.config.second_pair.trading_pair, TradeType.BUY, self.config.amount_quote_eth
            )
            executor_actions.extend(
                [
                    CreateExecutorAction(executor_config=btc_config, controller_id=self.config.id),
                    CreateExecutorAction(executor_config=eth_config, controller_id=self.config.id),
                ]
            )
            self._active_first_pair_short = True
            self._has_active_position = False

        return executor_actions

    async def update_processed_data(self):
        """Update any processed data needed by the controller"""
        if self._is_first_tick:
            self.logger().info("First tick detected, starting controller...")
            self._is_first_tick = False
            self.on_new_cycle_start()
            return None

        tick_current_time = int(time.time())

        self.check_and_handle_position_timeout(tick_current_time)

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

        if self._has_active_position:
            self._current_delta = self.calculate_current_relative_delta()
            self.check_and_handle_stop_loss()
            self.check_and_handle_take_profit()

        elif self.is_position_open_conditions():
            self.open_position()

        save_controller_state_to_file(self, tick_current_time)

        return None

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
            f"Has active position: {self._has_active_position}",
            f"Cumulative delta: {self.cumulative_delta}",
            f"Total positions opened count: {self._total_positions_opened_count}",
            f"Total positions closed count: {self._total_positions_closed_count}",
            f"Positions closed by timeout count: {self._positions_closed_by_timeout_count}",
            f"Positions closed by take profit count: {self._positions_closed_by_take_profit_count}",
            f"Positions closed by stop loss count: {self._positions_closed_by_stop_loss_count}",
            f"Current position opened at: {self._current_position_opened_at}",
            f"Reference delta updated at: {self._reference_delta_updated_at}",
            f"Reference prices updated at: {self._reference_prices_updated_at}",
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
        "has_active_position",
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
        "has_active_position": controller._has_active_position,
        "position_open_lower_threshold": controller.config.open_position_lower_threshold,
        "position_open_upper_threshold": controller.config.open_position_upper_threshold,
        "position_take_profit_threshold": controller.config.take_profit_threshold,
        "position_stop_loss_threshold": controller.config.stop_loss_threshold,
        "current_position_opened_at": controller._current_position_opened_at,
        "total_positions_opened_count": controller._total_positions_opened_count,
        "total_positions_closed_count": controller._total_positions_closed_count,
        "positions_closed_by_timeout_count": controller._positions_closed_by_timeout_count,
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
