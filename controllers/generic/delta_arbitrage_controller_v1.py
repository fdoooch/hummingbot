from decimal import Decimal
from typing import Dict, List, Set
from pydantic import Field, validator
import math

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.core.data_type.common import PriceType
from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction
from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType
from hummingbot.strategy_v2.executors.data_types import ConnectorPair
from hummingbot.client.config.config_data_types import ClientFieldData
import pandas as pd
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
    regression_interval: str = Field(
        default="1m",
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the regression interval (1m): ",
            prompt_on_new=True
        ))
    regression_window: int = Field(
        default=1000,
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the regression window size: ",
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
        default=5.5,
        client_data=ClientFieldData(
            prompt=lambda e: "Enter the close short threshold: ",
            prompt_on_new=True
        )
    )  # Exit BTC short/ETH long when ratio reaches 5.5


    @validator('candles_config', pre=True)
    def parse_candles_config(cls, v) -> List[CandlesConfig]:
        if isinstance(v, str):
            return cls.parse_candles_config_str(v)
        elif isinstance(v, list):
            return v
        raise ValueError("Invalid type for candles_config. Expected str or List[CandlesConfig]")


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
        self.pair_one_mid_price: float = 0.0
        self.pair_two_mid_price: float = 0.0
        self._current_ratio: float = 0.0
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
        df_one = self.market_data_provider.get_candles_df(connector_name=self.config.connector_one,
                                                          trading_pair=self.config.trading_pair_one,
                                                          interval=self.config.regression_interval,
                                                          max_records=self.config.regression_window)

        df_two = self.market_data_provider.get_candles_df(connector_name=self.config.connector_two,
                                                          trading_pair=self.config.trading_pair_two,
                                                          interval=self.config.regression_interval,
                                                          max_records=self.config.regression_window)

        pair_1_closing_prices = df_one["close"].values
        pair_2_closing_prices = df_two["close"].values
        self.regression_coef = self.calculate_regression_coef(pair_1_closing_prices, pair_2_closing_prices)



    def calculate_regression_coef(self, closing_prices_one: np.ndarray, closing_prices_two: np.ndarray) ->np.ndarray:
        # Log the lengths of the closing prices
        self.logger().info(f"Length of closing_prices_one: {len(closing_prices_one)}")
        self.logger().info(f"Length of closing_prices_two: {len(closing_prices_two)}")

        # Check if any of the arrays are empty
        if len(closing_prices_one) == 0 or len(closing_prices_two) == 0:
            self.logger().error("One or both of the closing prices arrays are empty. Skipping regression calculation.")
            return np.array([0, 0])  # Return default coefficients or handle as needed

        # Считаем линейную регрессию
        regression_coef: np.ndarray = np.polyfit(closing_prices_one, closing_prices_two, deg=1)
        return regression_coef


    def get_ratio(self, price_one: float, price_two: float) -> float:
        """Calculate current ratio"""
        pair_one = price_one * self.regression_coef[0] + self.regression_coef[1]
        pair_two = price_two
        ratio = pair_one / pair_two
        self.pair_one_normalized_price = pair_one
        return round(ratio, 3)
        

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

        # Получить свежие цены current_price_one и current_price_two
        self.pair_one_mid_price = float(self.market_data_provider.get_price_by_type(self.config.connector_one, self.config.trading_pair_one, PriceType.MidPrice))
        self.pair_two_mid_price = float(self.market_data_provider.get_price_by_type(self.config.connector_two, self.config.trading_pair_two, PriceType.MidPrice))

        if math.isnan(self.pair_one_mid_price) or math.isnan(self.pair_two_mid_price):
            return None
        
        self._current_ratio = self.get_ratio(self.pair_one_mid_price, self.pair_two_mid_price)
        ...
        # TODO: Расчёт сигналов на вход и выход
        signal_long = self._current_ratio <= self.config.open_long_threshold
        signal_short = self._current_ratio >= self.config.open_short_threshold

    def to_format_status(self) -> List[str]:
        """Format the current status for display"""
        status = [
            f"Pair 1 Price: {self.pair_one_mid_price}",
            f"Pair 2 Price: {self.pair_two_mid_price}",
            f"Ratio: {self._current_ratio}",
            f"Pair 1 Normalized Price: {self.pair_one_normalized_price}",
            f"Open Long Threshold: {self.config.open_long_threshold}",
            f"Open Short Threshold: {self.config.open_short_threshold}",
            f"Close Long Threshold: {self.config.close_long_threshold}",
            f"Close Short Threshold: {self.config.close_short_threshold}",
            f"Active Pair 1 Long: {self._active_first_pair_long}",
            f"Active Pair 1 Short: {self._active_first_pair_short}"
        ]
       
        return status