from decimal import Decimal
from typing import Dict, List, Set
from pydantic import Field, validator
import math
import time

from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.core.data_type.common import PriceType
from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction
from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType
from hummingbot.strategy_v2.executors.data_types import ConnectorPair
from hummingbot.client.config.config_data_types import ClientFieldData
import numpy as np


class RatioArbitrageControllerV1Config(ControllerConfigBase):
    controller_name: str = "ratio_arbitrage_controller_v1"
    candles_config: List[CandlesConfig] = Field(
        default="bybit_perpetual.BTC-USDT.1m.1000:bybit_perpetual.ETH-USDT.1m.1000:",
        client_data=ClientFieldData(
            is_updatable=True,
            prompt_on_new=True,
            prompt=lambda mi: (
                "Enter candle configs in format 'exchange1.tp1.interval1.max_records':"
                "'exchange2.tp2.interval2.max_records'"
            )
        )
    )
    connector_one: str = Field(
        default="bybit_perpetual",
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the connector ONE: ",
            prompt_on_new=True
        ))
    trading_pair_one: str = Field(
        default="BTC-USDT",
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the trading pair ONE: ",
            prompt_on_new=True
        ))
    connector_two: str = Field(
        default="bybit_perpetual",
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the connector TWO: ",
            prompt_on_new=True
        ))
    trading_pair_two: str = Field(
        default="ETH-USDT",
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the trading pair TWO: ",
            prompt_on_new=True
        ))
    delta_calculation_interval: str = Field(
        default="1m",
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the regression interval (1m): ",
            prompt_on_new=True
        ))
    reference_price_window: int = Field(
        default=100,
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the reference price window size: ",
            prompt_on_new=True
        ))

    # Ratio thresholds
    open_long_threshold: float = Field(
        default=5.0,
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the open long threshold: ",
            prompt_on_new=True
        )
    )  # UP_VALUE   Buy BTC/Sell ETH when BTC/ETH <= 5 
    open_short_threshold: float = Field(
        default=8.0,
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the open short threshold: ",
            prompt_on_new=True
        )
    )  # Sell BTC/Buy ETH when BTC/ETH >= 6
    close_long_threshold: float = Field(
        default=5.5,
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the close long threshold: ",
            prompt_on_new=True
        )
    )  # Exit BTC long/ETH short when ratio reaches 5.5
    close_short_threshold: float = Field(
        default=5,
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the close short threshold: ",
            prompt_on_new=True
        )
    )  # Exit BTC short/ETH long when ratio reaches 5.5
    fee_decimal: float = Field(
        default=0.001,
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the fee decimal: ",
            prompt_on_new=True
        )
    )


    @validator('candles_config', pre=True)
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
            entries = v.split(':')
            for entry in entries:
                parts = entry.split('.')
                if len(parts) != 4:
                    raise ValueError(f"Invalid candles config format in segment '{entry}'. "
                                     "Expected format: 'exchange.tradingpair.interval.maxrecords'")
                connector, trading_pair, interval, max_records_str = parts
                try:
                    max_records = int(max_records_str)
                except ValueError:
                    raise ValueError(f"Invalid max_records value '{max_records_str}' in segment '{entry}'. "
                                     "max_records should be an integer.")
                config = CandlesConfig(
                    connector=connector,
                    trading_pair=trading_pair,
                    interval=interval,
                    max_records=max_records
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


class RatioArbitrageControllerV1(ControllerBase):
    def __init__(self, config: RatioArbitrageControllerV1Config, *args, **kwargs):
        self.logger().warning(f"Initializing Ratio Arbitrage Controller with config: {config}")
        super().__init__(config, *args, **kwargs)
        self.config = config
        self._active_first_pair_long = False
        self._active_first_pair_short = False
        self._is_first_tick = True
        self.regression_coef: np.ndarray = np.array([])
        self.pair_one_current_price: float = 0.0
        self.pair_two_current_price: float = 0.0
        self._current_ratio: float = 0.0
        self.reference_price_one: float = 0.0
        self.reference_price_two: float = 0.0
        self._open_short_price_one: float = 0.0
        self._open_short_price_two: float = 0.0
        self._open_long_price_one: float = 0.0
        self._open_long_price_two: float = 0.0
        self._current_short_delta: float = 0.0
        self._reference_short_delta: float = 0.0
        self._current_long_delta: float = 0.0
        self._reference_long_delta: float = 0.0
        self.fee_decimal: float = float(config.fee_decimal)
        self.reference_updated_at: int = int(time.time())
        self.current_amount_quote: float = float(config.total_amount_quote)
        self.market_data_provider.initialize_rate_sources([ConnectorPair(
            connector_name=config.connector_one,
            trading_pair=config.trading_pair_one
        ), ConnectorPair(
            connector_name=config.connector_two,
            trading_pair=config.trading_pair_two
        )])


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
        # TODO: Получить 1000 минутных свечей для trading_pair_one и trading_pair_two
        # df_one = self.market_data_provider.get_candles_df(connector_name=self.config.connector_one,
        #                                                   trading_pair=self.config.trading_pair_one,
        #                                                   interval=self.config.delta_calculation_interval,
        #                                                   max_records=self.config.reference_price_window)

        # df_two = self.market_data_provider.get_candles_df(connector_name=self.config.connector_two,
        #                                                   trading_pair=self.config.trading_pair_two,
        #                                                   interval=self.config.delta_calculation_interval,
        #                                                   max_records=self.config.reference_price_window)

        # pair_1_closing_prices = df_one["close"].values
        # pair_2_closing_prices = df_two["close"].values
        self.update_reference_prices()
        ...



    def update_reference_prices(self):
        """
        Update historical prices for delta calculation
        """
        df_one = self.market_data_provider.get_candles_df(connector_name=self.config.connector_one,
                                                          trading_pair=self.config.trading_pair_one,
                                                          interval=self.config.delta_calculation_interval,
                                                          max_records=self.config.reference_price_window)

        df_two = self.market_data_provider.get_candles_df(connector_name=self.config.connector_two,
                                                          trading_pair=self.config.trading_pair_two,
                                                          interval=self.config.delta_calculation_interval,
                                                          max_records=self.config.reference_price_window)
        self.reference_price_one = df_one["close"].values[0]
        self.reference_price_two = df_two["close"].values[0]
        self.reference_updated_at = int(time.time())
        return None


    def calculate_delta_short(self, init_price_one: float, last_price_one: float, init_price_two: float, last_price_two: float, fee_decimal: float, amount_quote: float) -> float:

        pair_one_ratio = last_price_one / init_price_one
        pair_two_ratio = last_price_two / init_price_two

        delta = -amount_quote * (pair_one_ratio - pair_two_ratio)
        fee = fee_decimal * amount_quote * (pair_one_ratio + pair_two_ratio + 2)
        return delta - fee

    def is_open_short_conditions(self) -> bool:
        if self.reference_price_one == 0 or self.reference_price_two == 0 or self.pair_one_current_price == 0 or self.pair_two_current_price == 0:
            return False

        delta = self.calculate_delta_short(
            init_price_one=self.reference_price_one,
            last_price_one=self.pair_one_current_price,
            init_price_two=self.reference_price_two,
            last_price_two=self.pair_two_current_price,
            fee_decimal=self.fee_decimal,
            amount_quote=self.current_amount_quote
        )
        self._reference_short_delta = delta
        return delta > self.config.open_short_threshold


    def is_close_short_conditions(self) -> bool:

        delta = self.calculate_delta_short(
            init_price_one=self._open_short_price_one,
            last_price_one=self.pair_one_current_price,
            init_price_two=self._open_short_price_two,
            last_price_two=self.pair_two_current_price,
            fee_decimal=self.fee_decimal,
            amount_quote=self.current_amount_quote
        )
        self._current_short_delta = delta
        return delta > self.config.close_short_threshold
    

    def create_position_config(self, trading_pair: str, side: TradeType, amount_quote: Decimal, 
                               position_action: PositionAction = PositionAction.OPEN) -> PositionExecutorConfig:
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
        if self._active_first_pair_long and self._current_ratio >= self.config.exit_long_ratio:
            # Close BTC long and ETH short positions
            btc_config = self.create_position_config(
                self.config.first_pair.trading_pair,
                TradeType.SELL,  # Close long with sell
                self.config.amount_quote_btc,
                position_action=PositionAction.CLOSE
            )
            eth_config = self.create_position_config(
                self.config.second_pair.trading_pair,
                TradeType.BUY,   # Close short with buy
                self.config.amount_quote_eth,
                position_action=PositionAction.CLOSE
            )
            executor_actions.extend([
                CreateExecutorAction(executor_config=btc_config, controller_id=self.config.id),
                CreateExecutorAction(executor_config=eth_config, controller_id=self.config.id)
            ])
            self._active_first_pair_long = False

        elif self._active_first_pair_short and self._current_ratio <= self.config.exit_short_ratio:
            # Close BTC short and ETH long positions
            btc_config = self.create_position_config(
                self.config.first_pair.trading_pair,
                TradeType.BUY,   # Close short with buy
                self.config.amount_quote_btc,
                position_action=PositionAction.CLOSE
            )
            eth_config = self.create_position_config(
                self.config.second_pair.trading_pair,
                TradeType.SELL,  # Close long with sell
                self.config.amount_quote_eth,
                position_action=PositionAction.CLOSE
            )
            executor_actions.extend([
                CreateExecutorAction(executor_config=btc_config, controller_id=self.config.id),
                CreateExecutorAction(executor_config=eth_config, controller_id=self.config.id)
            ])
            self._active_first_pair_short = False

        # Check for entry conditions
        elif self._current_ratio <= self.config.lower_ratio and not self._active_first_pair_long:
            # Buy BTC and Sell ETH when ratio <= 5
            btc_config = self.create_position_config(
                self.config.first_pair.trading_pair,
                TradeType.BUY,
                self.config.amount_quote_btc
            )
            eth_config = self.create_position_config(
                self.config.second_pair.trading_pair,
                TradeType.SELL,
                self.config.amount_quote_eth
            )
            executor_actions.extend([
                CreateExecutorAction(executor_config=btc_config, controller_id=self.config.id),
                CreateExecutorAction(executor_config=eth_config, controller_id=self.config.id)
            ])
            self._active_first_pair_long = True
            self._active_first_pair_short = False

        elif self._current_ratio >= self.config.upper_ratio and not self._active_first_pair_short:
            # Sell BTC and Buy ETH when ratio >= 6
            btc_config = self.create_position_config(
                self.config.first_pair.trading_pair,
                TradeType.SELL,
                self.config.amount_quote_btc
            )
            eth_config = self.create_position_config(
                self.config.second_pair.trading_pair,
                TradeType.BUY,
                self.config.amount_quote_eth
            )
            executor_actions.extend([
                CreateExecutorAction(executor_config=btc_config, controller_id=self.config.id),
                CreateExecutorAction(executor_config=eth_config, controller_id=self.config.id)
            ])
            self._active_first_pair_short = True
            self._active_first_pair_long = False

        return executor_actions

    async def update_processed_data(self):
        """Update any processed data needed by the controller"""
        if self._is_first_tick:
            self.logger().info("First tick detected, starting controller...")
            self._is_first_tick = False
            self.on_new_cycle_start()
            return None

        time_since_reference = int(time.time()) - self.reference_updated_at
        if time_since_reference > 60:
            self.update_reference_prices()

        # Получить свежие цены current_price_one и current_price_two
        self.pair_one_current_price = float(self.market_data_provider.get_price_by_type(self.config.connector_one, self.config.trading_pair_one, PriceType.LastTrade))
        self.pair_two_current_price = float(self.market_data_provider.get_price_by_type(self.config.connector_two, self.config.trading_pair_two, PriceType.LastTrade))

        if self._active_first_pair_short:
            if self.is_close_short_conditions():
                self.exit_short()

        elif self.is_open_short_conditions():
            self.open_short()

        return None



    def open_short(self):
        self._open_short_price_one = self.pair_one_current_price
        self._open_short_price_two = self.pair_two_current_price
        self._active_first_pair_short = True
        self._active_first_pair_long = False

    def exit_short(self):
        self._open_short_price_one = 0.0
        self._open_short_price_two = 0.0
        self._current_short_delta = 0.0
        self._active_first_pair_short = False


    def to_format_status(self) -> List[str]:
        """Format the current status for display"""
        status = [
            f"Pair 1 Price: {self.pair_one_current_price}",
            f"Reference Price One: {self.reference_price_one}",
            f"Pair 2 Price: {self.pair_two_current_price}",
            f"Reference Price Two: {self.reference_price_two}",
            f"Reference Updated At: {self.reference_updated_at}",
            f"Short Delta: {self._current_short_delta}",
            f"Reference Short Delta: {self._reference_short_delta}",
            f"Long Delta: {self._current_long_delta}",
            f"Reference Long Delta: {self._reference_long_delta}",
            # f"Pair 1 Normalized Price: {self.pair_one_normalized_price}",
            f"Open Long Threshold: {self.config.open_long_threshold}",
            f"Open Short Threshold: {self.config.open_short_threshold}",
            f"Close Long Threshold: {self.config.close_long_threshold}",
            f"Close Short Threshold: {self.config.close_short_threshold}",
            f"Active Pair 1 Long: {self._active_first_pair_long}",
            f"Active Pair 1 Short: {self._active_first_pair_short}"
        ]
       
        return status