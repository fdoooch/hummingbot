from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Set
from pydantic import BaseModel, Field, validator
import math

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import OrderType, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.models.base import RunnableStatus

import os
import csv
import time


"""
Вход в позиции осуществляется по DCA.

"""


class EntryBucket(BaseModel):
    id: str
    drawdown_percentage: float  # На сколько процентов должна упасть цена от средней для входа.
    amount: float


class ExitBucket(BaseModel):
    id: str
    profit_percentage: float  # на сколько процентов должна вырасти цена от средней для выхода.
    qty_share: float


class GridBucketStatus(Enum):
    NEW = "NEW"
    NOT_ACTIVE = "NOT_ACTIVE"
    ACTIVE = "ACTIVE"
    PLACED = "PLACED"
    FILLED = "FILLED"
    SKIPPED = "SKIPPED"  # in case small amount


class GridBucket(BaseModel):
    id: str
    price: float
    qty: float
    amount: float
    side: TradeType
    status: GridBucketStatus = GridBucketStatus.NOT_ACTIVE
    is_last: bool = False
    is_market: bool = False


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
        return self.status in [RunnableStatus.TERMINATED, RunnableStatus.SHUTTING_DOWN]


class BucketsV2ControllerConfig(ControllerConfigBase):
    controller_name = "buckets_v2"
    controller_type: str = "generic"
    connector_name: str = Field(
        default="binance_perpetual",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the name of the exchange to trade on (e.g., binance_perpetual):",
        ),
    )
    trading_pair: str = Field(
        default="BTC-USDT",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the trading pair to trade on (e.g., BTC-USDT):",
        ),
    )
    candles_config: List[CandlesConfig] = []
    entry_side: TradeType = Field(
        default="BUY",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the entry side (BUY or SELL): ",
        ),
    )
    entry_order_type: OrderType = Field(
        default="MARKET",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the entry orders type (MARKET or LIMIT): ",
        ),
    )
    exit_order_type: OrderType = Field(
        default="MARKET",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the exit orders type (MARKET or LIMIT): ",
        ),
    )
    position_mode: PositionMode = Field(
        default="ONEWAY",
    )
    leverage: int = Field(
        default=1,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Set the leverage to use for trading (e.g., 20 for 20x leverage). Set it to 1 for spot trading: ",
        ),
    )
    entry_buckets: List[EntryBucket] = Field(
        default=[EntryBucket(id="entry_0", drawdown_percentage=0.0, amount=0.0)],
        client_data=ClientFieldData(
            is_updatable=True,
        ),
    )
    exit_buckets: List[ExitBucket] = Field(
        default=[ExitBucket(id="exit_0", profit_percentage=0.5, qty_share=1.0)],
        client_data=ClientFieldData(
            is_updatable=True,
        ),
    )
    activation_bounds: float = Field(default=0.01, client_data=ClientFieldData(is_updatable=True))
    min_order_amount: float = Field(
        default=10.0,
        client_data=ClientFieldData(prompt_on_new=True, is_updatable=True),
        prompt=lambda mi: "Enter the minimum order amount in USDT: ",
    )
    trailing_entry: bool = Field(
        default=False,
        client_data=ClientFieldData(prompt_on_new=True, is_updatable=True),
        prompt=lambda mi: "Trail entry level after the price: ",
    )

    @validator("entry_side", pre=True, allow_reuse=True)
    def validate_entry_side(cls, v) -> TradeType:
        if isinstance(v, TradeType):
            return v
        elif v is None:
            return TradeType.BUY
        elif isinstance(v, str):
            if v.upper() in TradeType.__members__:
                return TradeType[v.upper()]
        elif isinstance(v, int):
            try:
                return TradeType(v)
            except ValueError:
                pass
        raise ValueError(f"Invalid order type: {v}. Valid options are: {', '.join(TradeType.__members__)}")

    @validator("entry_order_type", pre=True, allow_reuse=True)
    def validate_entry_order_type(cls, v) -> OrderType:
        if isinstance(v, OrderType):
            return v
        elif v is None:
            return OrderType.MARKET
        elif isinstance(v, str):
            if v.upper() in OrderType.__members__:
                return OrderType[v.upper()]
        elif isinstance(v, int):
            try:
                return OrderType(v)
            except ValueError:
                pass
        raise ValueError(f"Invalid order type: {v}. Valid options are: {', '.join(OrderType.__members__)}")

    @validator("exit_order_type", pre=True, allow_reuse=True)
    def validate_exit_order_type(cls, v) -> OrderType:
        if isinstance(v, OrderType):
            return v
        elif v is None:
            return OrderType.MARKET
        elif isinstance(v, str):
            if v.upper() in OrderType.__members__:
                return OrderType[v.upper()]
        elif isinstance(v, int):
            try:
                return OrderType(v)
            except ValueError:
                pass
        raise ValueError(f"Invalid order type: {v}. Valid options are: {', '.join(OrderType.__members__)}")

    @validator("position_mode", pre=True, allow_reuse=True)
    def validate_position_mode(cls, v) -> PositionMode:
        if isinstance(v, PositionMode):
            return v
        elif v is None:
            return PositionMode.ONEWAY
        elif isinstance(v, str):
            if v.upper() in PositionMode.__members__:
                return PositionMode[v.upper()]
        elif isinstance(v, int):
            try:
                return PositionMode(v)
            except ValueError:
                pass
        raise ValueError(f"Invalid position mode: {v}. Valid options are: {', '.join(PositionMode.__members__)}")

    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        if self.connector_name not in markets:
            markets[self.connector_name] = set()
        markets[self.connector_name].add(self.trading_pair)
        return markets


class BucketsV2Controller(ControllerBase):
    inits_count: int = 0
    cycle_num: int = 0

    @property
    def state(self) -> dict[str, Any]:
        return {}

    @property
    def executors_states(self) -> dict[str, ExecutorState]:
        executors_states = {ex.custom_info["level_id"]: ExecutorState(**ex.to_dict()) for ex in self.executors_info}
        return executors_states

    @property
    def has_inactive_exit_bucket(self) -> bool:
        for bucket in self.exit_buckets.values():
            if bucket.status == GridBucketStatus.NOT_ACTIVE:
                return True
        return False

    def __init__(self, config: BucketsV2ControllerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.config.entry_buckets = self.sort_config_entry_buckets(self.config.entry_buckets, self.config.entry_side)
        self.config.exit_buckets = self.sort_config_exit_buckets(self.config.exit_buckets, self.config.entry_side)
        self.position_mode = self.config.position_mode
        self.entry_side = self.config.entry_side
        self.exit_side = TradeType.SELL if self.entry_side == TradeType.BUY else TradeType.BUY
        self.entry_buckets: dict[str, GridBucket] = {}
        self.exit_buckets: dict[str, GridBucket] = {}
        self._entry_grid_updated_at = 0
        self._exit_grid_updated_at = 0
        self.current_qty: float = 0.0
        self.start_price: float = self.get_last_trade_price()
        self.last_trade_price = self.start_price
        self.avg_entry_price: float = self.start_price
        self.executors_actions: List[ExecutorAction] = []
        self.trading_rules = None
        self.entry_buckets: dict[str, GridBucket] = {}
        self.exit_buckets: dict[str, GridBucket] = {}
        self.trades: list = []
        self.is_active = True
        self.is_last_exit_bucket_filled = False
        self._is_new_bucket_filled = False
        self._is_new_entry_bucket_filled = False
        self._is_new_exit_bucket_filled = False
        BucketsV2Controller.inits_count += 1
        self.logger().info(f"INIT -> Controller initialized: {BucketsV2Controller.inits_count}")

    def quantize_price(self, price: float) -> float:
        price_quantized = self.market_data_provider.quantize_order_price(
            self.config.connector_name, self.config.trading_pair, Decimal(str(price))
        )
        price_quantized_float = float(price_quantized)
        if math.isnan(price_quantized_float):
            self.logger().warning("Quantized price is NaN. Using original price: %s", price)
            price_quantized_float = price
        return price_quantized_float

    def quantize_qty(self, qty: float) -> float:
        qty_quantized = self.market_data_provider.quantize_order_amount(
            self.config.connector_name, self.config.trading_pair, Decimal(str(qty))
        )
        qty_quantized_float = float(qty_quantized)
        if math.isnan(qty_quantized_float):
            self.logger().warning("Quantized qty is NaN. Using original qty: %s", qty)
            qty_quantized_float = qty
        return qty_quantized_float

    def sort_config_entry_buckets(self, buckets: list[EntryBucket], trade_side: TradeType) -> list[EntryBucket]:
        return buckets

    def sort_config_exit_buckets(self, buckets: list[ExitBucket], trade_side: TradeType) -> list[ExitBucket]:
        return sorted(buckets, key=lambda x: x.profit_percentage)

    def _calculate_entry_buckets(self) -> Dict[str, GridBucket]:
        self.logger().info("calculate_entry_buckets")
        # self.logger().info("Entry buckets: %s", self.entry_buckets)
        self.trading_rules = self.market_data_provider.get_trading_rules(
            self.config.connector_name, self.config.trading_pair
        )
        buckets = {}
        min_amount = self.config.min_order_amount

        for bucket in self.config.entry_buckets:
            bucket_id = f"{self.cycle_num}_{bucket.id}"
            if self.entry_buckets.get(bucket_id) and self.entry_buckets[bucket_id].status == GridBucketStatus.FILLED:
                buckets[bucket_id] = self.entry_buckets[bucket_id]
                continue

            if self.entry_side == TradeType.BUY:
                price = self.avg_entry_price * (1 - bucket.drawdown_percentage / 100)
            else:
                price = self.avg_entry_price * (1 + bucket.drawdown_percentage / 100)

            price_quantized = self.quantize_price(price)
            qty = bucket.amount * self.config.leverage / price_quantized
            qty_quantized = self.quantize_qty(qty)
            amount = qty_quantized * price_quantized
            status = GridBucketStatus.NOT_ACTIVE if amount >= min_amount else GridBucketStatus.SKIPPED

            buckets[bucket_id] = GridBucket(
                id=bucket_id,
                price=price_quantized,
                qty=qty_quantized,
                amount=amount,
                side=self.entry_side,
                status=status,
                is_market=True if bucket.drawdown_percentage == 0 else False,
            )
            self.logger().info("BUCKET: %s", buckets[bucket_id])
        return buckets

    def _calculate_exit_buckets(self) -> Dict[str, GridBucket]:
        self.logger().info(f"calculate_exit_buckets. Exit side: {self.exit_side}")
        buckets = {}
        # self.logger().info("current_qty: %s", self.current_qty)
        if self.current_qty == 0:
            return buckets

        min_amount = self.config.min_order_amount
        available_qty = self.current_qty
        average_entry_price = self.avg_entry_price

        for bucket in self.config.exit_buckets:
            bucket_id = f"{self.cycle_num}_{bucket.id}"
            # if self.exit_buckets.get(bucket_id) and self.exit_buckets[bucket_id].status == GridBucketStatus.FILLED:
            #     buckets[bucket_id] = self.exit_buckets[bucket_id]
            #     continue

            if self.exit_side == TradeType.BUY:
                bucket_price = average_entry_price * (1 - bucket.profit_percentage / 100)
            else:
                bucket_price = average_entry_price * (1 + bucket.profit_percentage / 100)
            price_quantized = self.quantize_price(bucket_price)

            qty = bucket.qty_share * available_qty
            qty_quantized = self.quantize_qty(qty)
            amount = qty_quantized * price_quantized
            if amount >= min_amount:
                status = GridBucketStatus.NOT_ACTIVE
                available_qty -= qty_quantized
            else:
                status = GridBucketStatus.SKIPPED

            buckets[bucket_id] = GridBucket(
                id=bucket_id,
                price=price_quantized,
                qty=qty_quantized,
                amount=amount,
                side=self.exit_side,
                status=status,
            )
            self.logger().info("BUCKET: %s", buckets[bucket_id])
            # self.logger().debug("available_qty: %s", available_qty)
            # self.logger().info("buckets: %s", buckets)
        if available_qty > 0:
            self.logger().warning("Not all QTY was allocated to exit buckets. Available QTY: %s", available_qty)

        # Determine the last exit_bucket
        available_qty = 0
        for bucket_id, bucket in reversed(buckets.items()):
            if bucket.status == GridBucketStatus.FILLED:
                continue
            if bucket.status == GridBucketStatus.SKIPPED:
                if (bucket.qty + available_qty) * bucket.price < min_amount:
                    available_qty += bucket.qty
                    continue
                else:
                    qty = bucket.qty + available_qty
                    buckets[bucket_id].qty = qty
                    buckets[bucket_id].amount = qty * bucket.price
                    buckets[bucket_id].status = GridBucketStatus.NOT_ACTIVE
                    buckets[bucket_id].is_last = True
                    break

            qty = bucket.qty + available_qty
            buckets[bucket_id].qty = qty
            buckets[bucket_id].amount = qty * bucket.price
            buckets[bucket_id].status = GridBucketStatus.NOT_ACTIVE
            buckets[bucket_id].is_last = True
            break
        return buckets

    def _handle_executors_updates(self, grid_buckets: List[GridBucket]):
        executors_states = self.executors_states
        result_buckets = {}

        for bucket_id, bucket in grid_buckets.items():
            if not executors_states.get(bucket_id):
                result_buckets[bucket_id] = bucket
                continue
            if bucket.status == GridBucketStatus.FILLED:
                result_buckets[bucket_id] = bucket
                continue

            if executors_states[bucket_id].is_active_and_not_filled:
                bucket.status = GridBucketStatus.ACTIVE

            elif executors_states[bucket_id].is_full_filled:
                self.logger().info(f"BUCKET FILLED {bucket_id}: {bucket}")
                # self.logger().info(f"Executors: {self.executors_states}")
                base_symbol = self.config.trading_pair.split("-")[0]
                exchange_qty = float(self.market_data_provider.get_balance(self.config.connector_name, base_symbol))
                # self.logger().info(f"Exchange QTY: {exchange_qty}")
                # self.logger().info(f"Current QTY: {self.current_qty}\n\n")

                self._is_new_bucket_filled = True
                calculated_qty = self.current_qty + (
                    executors_states[bucket_id].qty
                    if bucket.side == TradeType.BUY
                    else -executors_states[bucket_id].qty
                )
                # Get the base symbol from the trading pair (e.g., 'BTC' from 'BTC-USDT')
                base_symbol = self.config.trading_pair.split("-")[0]
                exchange_qty = float(self.market_data_provider.get_balance(self.config.connector_name, base_symbol))
                self.current_qty = min(calculated_qty, exchange_qty)

                bucket.status = GridBucketStatus.FILLED
                bucket.price = executors_states[bucket_id].average_entry_price
                bucket.qty = executors_states[bucket_id].qty
                bucket.amount = executors_states[bucket_id].amount
                if bucket.is_last:
                    self.is_last_exit_bucket_filled = True

            result_buckets[bucket_id] = bucket
        # self.logger().info(f"handle executors updates.\nENTRY_BUCKETS: {self.entry_buckets}\nEXIT_BUCKETS: {self.exit_buckets}\ngrid_buckets: {grid_buckets}\nresult_buckets: {result_buckets}")

        return result_buckets

    def get_current_qty_from_exchange(self) -> float:
        # Get the base symbol from the trading pair (e.g., 'BTC' from 'BTC-USDT')
        base_symbol = self.config.trading_pair.split("-")[0]
        exchange_qty = self.market_data_provider.get_balance(self.config.connector_name, base_symbol)
        if math.isnan(exchange_qty):
            return 0.0
        return float(exchange_qty)

    def _is_price_out_of_trading_range(self, price: float, side: TradeType) -> bool:
        if side == TradeType.BUY:
            return price <= self.config.activation_bounds
        elif side == TradeType.SELL:
            return price >= self.config.activation_bounds
        else:
            raise ValueError(f"Unknown trade type: {side}")

    def get_mid_price(self) -> float:
        mid_price = self.market_data_provider.get_price_by_type(
            self.config.connector_name, self.config.trading_pair, PriceType.MidPrice
        )
        if math.isnan(mid_price):
            return 0.0
        return float(mid_price)

    def _on_last_exit_bucket_filled(self) -> bool:
        self.logger().info("LAST EXIT BUCKET FILLED")
        stop_actions = [
            StopExecutorAction(controller_id=self.config.id, executor_id=executor.id, keep_position=True)
            for executor in self.executors_info
        ]
        self.executors_actions.extend(stop_actions)
        self.cycle_num += 1
        self.reset_buckets()

    # def fetch_start_price(self) -> Decimal:
    #     while True:
    #         try:
    #             price = self.get_last_trade_price()
    #             self.logger().info("Start price fetching: %s", price)
    #             if price:
    #                 return price
    #             self.logger().info("Waiting for start price...")
    #             time.sleep(1)
    #         except ValueError:
    #             self.logger().info("VALUE ERROR: Waiting for start price...")
    #             time.sleep(1)
    #         except Exception as e:
    #             self.logger().error("Error fetching start price: %s", e)
    #             time.sleep(1)

    def get_last_trade_price(self) -> float:
        self.last_trade_price = float(
            self.market_data_provider.get_price_by_type(
                self.config.connector_name,
                self.config.trading_pair,
                PriceType.LastTrade,
            )
        )
        if math.isnan(self.last_trade_price):
            self.last_trade_price = 0.0

        return self.last_trade_price

    def active_executors(self, is_trading: bool) -> List[ExecutorInfo]:
        return [
            executor for executor in self.executors_info if executor.is_active and executor.is_trading == is_trading
        ]

    def calculate_average_long_entry_price(self) -> float:
        price = 0.0
        qty = 0.0
        amount = 0.0
        for bucket_id, bucket in self.entry_buckets.items():
            if bucket.status == GridBucketStatus.FILLED:
                qty += bucket.qty
                amount += bucket.amount
        if qty > 0:
            price = amount / qty
        if price != 0.0:
            # self.logger().info(f"AVERAGE_ENTRY_PRICE: {price}")
            return price
        # self.logger().info(f"AVERAGE_ENTRY_PRICE is START_PRICE: {self.start_price}")
        return self.start_price

    async def update_processed_data(self):
        """
        Update the processed data based on the current state of the strategy.
        """
        if math.isnan(self.start_price) or self.start_price == 0.0:
            self.logger().info("START_PRICE is NaN. Fetching last trade price...")
            self.start_price = self.get_last_trade_price()
            if math.isnan(self.avg_entry_price) or self.avg_entry_price == 0.0:
                self.avg_entry_price = self.start_price
            return None

        mid_price = self.get_mid_price()
        last_trade_price = self.get_last_trade_price()
        if not self.is_active:
            self.processed_data.update(
                {
                    "start_price": self.start_price,
                    "mid_price": mid_price,
                    "last_trade_price": last_trade_price,
                }
            )
            return None

        if self._is_new_entry_bucket_filled:
            self.exit_buckets = self._calculate_exit_buckets()
            self._is_new_entry_bucket_filled = False

        if self._is_new_exit_bucket_filled:
            ...
            self._is_new_exit_bucket_filled = False

        avg_entry_price = self.avg_entry_price
        self.entry_buckets = self._handle_executors_updates(self.entry_buckets)
        # self.logger().info(f"ENTRY BUCKETS after handle executors: {self.entry_buckets}")
        self._is_new_entry_bucket_filled = self._is_new_bucket_filled
        self._is_new_bucket_filled = False

        if self._is_new_entry_bucket_filled:
            self.avg_entry_price = self.calculate_average_long_entry_price()

        self.exit_buckets = self._handle_executors_updates(self.exit_buckets)
        self._is_new_exit_bucket_filled = self._is_new_bucket_filled
        self._is_new_bucket_filled = False

        if avg_entry_price != self.avg_entry_price:
            self.logger().info(f"AVERAGE entry price updated to {self.avg_entry_price}")

        if self._is_new_entry_bucket_filled:
            self.entry_buckets = self._calculate_entry_buckets()
            self.exit_buckets = self._calculate_exit_buckets()

        if self.is_last_exit_bucket_filled:
            self._on_last_exit_bucket_filled()

        # avg_entry_price = self.calculate_average_long_entry_price()
        self.processed_data.update(
            {
                "mid_price": mid_price,
                "last_trade_price": last_trade_price,
                "avg_entry_price": avg_entry_price,
                # "active_executors_order_placed": self.active_executors(is_trading=False),
                # "active_executors_order_trading": self.active_executors(is_trading=True),
                "long_activation_bounds": mid_price * (1 - self.config.activation_bounds),
                "short_activation_bounds": mid_price * (1 + self.config.activation_bounds),
            }
        )
        # save_controller_state_to_file(self)

    def _determine_entry_executors(self):
        actions = []
        executors_states = self.executors_states
        # self.logger().info(f"DETERMINE ENTRY EXECUTORS ({len(executors_states)})")
        # self.logger().info(f"Executors: {self.executors_info}")
        for bucket_id, bucket in self.entry_buckets.items():
            executor_state = executors_states.get(bucket_id)

            if bucket.status == GridBucketStatus.FILLED:
                if executor_state and not executor_state.is_terminated:
                    # self.logger().info(f"STOP FILLED ENTRY EXECUTOR. {bucket}")
                    # self.logger().info(f"EXECUTOR STATE: {executor_state}")
                    actions.append(
                        StopExecutorAction(
                            controller_id=self.config.id,
                            executor_id=executor_state.id,
                            keep_position=True,
                        )
                    )
                continue

            if (
                bucket.status == GridBucketStatus.ACTIVE
                and executor_state.is_active_and_not_filled
                and executor_state.qty == bucket.qty
                and executor_state.config.entry_price == bucket.price
            ):
                # Place only one entry order at once
                break

            # stop current bucket executor
            if executors_states.get(bucket_id):
                self.logger().info(f"STOP ENTRY EXECUTOR. {bucket}")
                self.logger().info(f"EXECUTOR STATE: {executors_states[bucket_id]}")
                actions.append(
                    StopExecutorAction(
                        controller_id=self.config.id,
                        executor_id=executors_states[bucket_id].id,
                        keep_position=False,
                    )
                )

            if bucket.status in [GridBucketStatus.SKIPPED]:
                continue

            executor_config = (
                self.get_market_entry_executor_config(
                    qty=bucket.qty,
                    level_id=bucket.id,
                )
                if bucket.is_market
                else self.get_limit_entry_executor_config(
                    price=bucket.price,
                    qty=bucket.qty,
                    level_id=bucket.id,
                )
            )

            # create new bucket executor
            self.logger().info(f"NEW ENTRY EXECUTOR. {bucket}")
            actions.append(
                CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=executor_config,
                ),
            )
            # place only one entry order per once
            break
        self.logger().info(f"ENTRY EXECUTOR ACTIONS: {actions}")
        return actions

    def _determine_exit_executors(self):
        actions = []
        executors_states = self.executors_states
        for bucket_id, bucket in self.exit_buckets.items():
            executor_state = executors_states.get(bucket_id)

            if bucket.status == GridBucketStatus.FILLED:
                if executor_state and not executor_state.is_terminated:
                    self.logger().info(f"STOP FILLED EXIT EXECUTOR. {bucket}")
                    self.logger().info(f"EXECUTOR STATE: {executor_state}")
                    actions.append(
                        StopExecutorAction(
                            controller_id=self.config.id,
                            executor_id=executor_state.id,
                            keep_position=True,
                        )
                    )
                continue

            if (
                bucket.status == GridBucketStatus.ACTIVE
                and executor_state.is_active_and_not_filled
                and executor_state.config_qty == bucket.qty
                and executor_state.config_entry_price == bucket.price
            ):
                # Place only one entry order at once
                continue

            # stop current bucket executor
            if executor_state:
                self.logger().info(f"STOP EXIT EXECUTOR. {bucket}")
                self.logger().info(f"EXECUTOR STATE: {executor_state}")
                self.logger().info(
                    f"EXECUTOR => is_active_and_not_filled: {executor_state.is_active_and_not_filled}, qty: {executor_state.config_qty} [{type(executor_state.config_qty)}] ({executor_state.config_qty == bucket.qty}), price: {executor_state.config_entry_price} [{type(executor_state.config_entry_price)}] ({executor_state.config_entry_price == bucket.price}), bucket_qty: {bucket.qty} [{type(bucket.qty)}], bucket_price: {bucket.price} [{type(bucket.price)}]"
                )
                actions.append(
                    StopExecutorAction(
                        controller_id=self.config.id,
                        executor_id=executor_state.id,
                        keep_position=False,
                    )
                )

            if bucket.status == GridBucketStatus.SKIPPED:
                continue

            if bucket.side == TradeType.BUY:
                price = min(self.last_trade_price, bucket.price)

            else:
                price = max(self.last_trade_price, bucket.price)

            # create new bucket executor
            self.logger().info(f"NEW EXIT EXECUTOR. {bucket}")
            # executors_str = "\n".join([f"{k}: {v}" for k, v in executors_states.items()])
            # self.logger().info(f"Executors:\n{executors_str}")
            actions.append(
                CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=self.get_exit_executor_config(
                        price=price,
                        qty=bucket.qty,
                        level_id=bucket.id,
                    ),
                )
            )
            # create one executor per tick
            break

        return actions

    def stop_all_active_exit_executors(self):
        actions = []
        executors_states = self.executors_states
        for bucket_id, bucket in self.exit_buckets.items():
            executor_state = executors_states.get(bucket_id)
            if executor_state and not executor_state.is_terminated:
                self.logger().info(f"STOP EXIT EXECUTOR (stop all). {bucket}")
                self.logger().info(f"EXECUTOR STATE: {executor_state}")
                actions.append(
                    StopExecutorAction(
                        controller_id=self.config.id,
                        executor_id=executor_state.id,
                        keep_position=executor_state.is_full_filled,
                    )
                )

        return actions


    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        Determine actions based on the provided executor handler report.
        """
        # self.logger().info(f"Determin executor actions. start_price: {self.start_price}")
        if math.isnan(self.start_price) or self.start_price == 0.0 or self.avg_entry_price == 0.0:
            return []

        actions = self.executors_actions
        if self.entry_buckets == {}:
            # первый тик нового цикла
            self.entry_buckets = self._calculate_entry_buckets()
            actions.extend(self._determine_entry_executors())

        # actions.extend(self.determine_create_executor_actions())
        # actions.extend(self.determine_stop_executor_actions())
        if self._is_new_entry_bucket_filled:
            actions.extend(self.stop_all_active_exit_executors())
            actions.extend(self._determine_entry_executors())
 

        elif self._is_new_exit_bucket_filled or self.has_inactive_exit_bucket:
            actions.extend(self._determine_exit_executors())

        self.executors_actions = []
        return actions

    # def determine_create_executor_actions(self) -> List[ExecutorAction]:
    #     create_actions = []

    #     if self.can_create_entry_orders_executor():
    #         create_actions.extend(self._update_entry_executors())
    #         for bucket_id, bucket in self.entry_buckets.items():
    #             if bucket.status == GridBucketStatus.ACTIVE:
    #                 break

    #             if bucket.status == GridBucketStatus.NOT_ACTIVE:
    #                 create_actions.append(
    #                     CreateExecutorAction(
    #                         controller_id=self.config.id,
    #                         executor_config=self.get_limit_entry_executor_config(
    #                             price=bucket.price,
    #                             qty=bucket.qty,
    #                             level_id=bucket.id,
    #                         ),
    #                     )
    #                 )
    #                 break

    #     return create_actions

    def can_create_entry_orders_executor(self) -> bool:
        # return len(self.executors_info) == 0
        return True

    def can_create_exit_orders_executor(self) -> bool:
        return True

    # def determine_stop_executor_actions(self) -> List[ExecutorAction]:
    #     stop_actions: List[StopExecutorAction] = []
    #     return stop_actions

    def get_limit_entry_executor_config(self, price: float, qty: float, level_id: int) -> PositionExecutorConfig:
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=self.config.entry_side,
            entry_price=price,
            amount=qty,
            leverage=self.config.leverage,
            activation_bounds=[
                self.config.activation_bounds,
                self.config.activation_bounds,
            ],
            level_id=level_id,
            position_mode=self.position_mode,
        )

    def get_market_entry_executor_config(self, qty: float, level_id: int) -> PositionExecutorConfig:
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=self.config.entry_side,
            amount=qty,
            leverage=self.config.leverage,
            activation_bounds=[
                self.config.activation_bounds,
                self.config.activation_bounds,
            ],
            level_id=level_id,
            position_mode=self.position_mode,
            triple_barrier_config=TripleBarrierConfig(
                open_order_type=OrderType.MARKET,
            ),
        )

    def get_exit_executor_config(self, price: float, qty: float, level_id: int) -> PositionExecutorConfig:
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=TradeType.SELL if self.config.entry_side == TradeType.BUY else TradeType.BUY,
            entry_price=price,
            amount=qty,
            leverage=self.config.leverage,
            activation_bounds=[
                self.config.activation_bounds,
                self.config.activation_bounds,
            ],
            level_id=level_id,
            position_mode=self.position_mode,
        )

    def reset_buckets(self):
        self.logger().info("RESETTING buckets")
        self.current_qty = 0.0
        self.start_price = self.get_last_trade_price()
        self.avg_entry_price = self.start_price
        self.entry_buckets = {}
        self.exit_buckets = {}
        self._is_new_entry_bucket_filled = False
        self._is_new_exit_bucket_filled = False
        self.is_last_exit_bucket_filled = False

    def on_stop(self):
        self.logger().info("ON STOP STARTED")
        executors_states = self.executors_states
        stop_actions = []
        for executor_id, executor_state in executors_states.items():
            self.logger().info(f"STOP EX. {executor_id} - {executor_state}")
            keep_position = not executor_state.is_active_and_not_filled
            stop_actions.append(
                StopExecutorAction(
                    controller_id=self.config.id, executor_id=executor_state.id, keep_position=keep_position
                )
            )
        self.executors_actions.extend(stop_actions)
        self.is_active = False
        self.logger().info("ON STOP FINISHED")

    def to_format_status(self) -> List[str]:
        # Check if there are any buckets before creating the table
        if not self.entry_buckets and not self.exit_buckets:
            table_string = "No buckets available"
        else:
            table_string = self.create_buckets_table_string(self.entry_buckets | self.exit_buckets)

        active_executors = self.active_executors(is_trading=True)  # те, чьи ордера исполнены
        active_executors_placed = self.active_executors(is_trading=False)
        return [
            f"Long activation bounds: {self.processed_data['long_activation_bounds']}",
            f"Short activation bounds: {self.processed_data['short_activation_bounds']}",
            f"Total executors count: {len(self.executors_info)}",
            f"Active executors count: {len(active_executors)}",
            f"Active executors order placed count: {len(active_executors_placed)}",
            f"Active executors:\n{self.create_executors_string(active_executors)}",
            f"Active executors placed:\n{self.create_executors_string(active_executors_placed)}",
            f"ALL executors:\n{self.create_executors_string(self.executors_info)}",
            f"Buckets:\n{table_string}",
            f"Is active: {self.is_active}",
            f"Start price: {self.start_price}",
            f"Mid price: {self.processed_data['mid_price']}",
            f"\nLast trade price: {self.processed_data['last_trade_price']}",
            f"Average entry price: {self.processed_data['avg_entry_price']}",
            f"Current qty: {self.current_qty}",
            f"Minimal order amount: {self.config.min_order_amount}",
            # f"Placed details:\n{active_executors_placed}"
        ]

    def create_executors_string(self, executors: List[ExecutorInfo]) -> str:
        result = ""
        for executor in executors:
            result += f""">> {executor.id}
    Close type: {executor.close_type}
    Status: {executor.status}
    Entry price: {executor.config.entry_price}
    Qty: {executor.config.amount}
    Filled amount: {executor.filled_amount_quote}
    Is trading: {executor.is_trading}
    Is active: {executor.is_active}
    # Activation bounds: [{executor.config.entry_price * (1 - executor.config.activation_bounds[0])}, {executor.config.entry_price * (1 + executor.config.activation_bounds[1])}]
    # Activation bounds: {executor.config.activation_bounds}
    ID: {executor.custom_info.get("level_id")}
    Order IDs: {executor.custom_info.get("order_ids")}
    \n
    """
        return result

    def create_buckets_table_string(self, buckets: Dict[str, GridBucket]):
        # Column headers
        headers = ["ID", "STATUS", "PRICE", "QTY", "AMOUNT", "IS_LAST"]

        # Collect all data in a single pass and calculate max widths simultaneously
        bucket_data = [
            (
                str(b.id),
                str(b.status.value),
                str(b.price),
                str(b.qty),
                str(b.price * b.qty),
                "*" if b.is_last else "",
            )
            for i, b in buckets.items()
        ]

        # If there are no buckets, return a message instead of trying to create a table
        if not bucket_data:
            return "No buckets available"

        # Calculate column widths - compare header with longest data in each column
        col_widths = [max(len(header), max(len(row[i]) for row in bucket_data)) + 2 for i, header in enumerate(headers)]

        # Create separator and header row once
        separator = "+" + "+".join("-" * width for width in col_widths) + "+"
        header_row = "|" + "|".join(f"{header:^{width}}" for header, width in zip(headers, col_widths)) + "|"

        # Create data rows in one pass
        data_rows = [
            "|" + "|".join(f"{value:^{width}}" for value, width in zip(row, col_widths)) + "|" for row in bucket_data
        ]

        # Join all parts
        return "\n".join([separator, header_row, separator] + data_rows + [separator])


def save_controller_state_to_file(controller: BucketsV2Controller):
    filename = f"logs/{controller.config.id}.csv"

    headers = [
        "timestamp",
        "cycle_num",
        "current_qty",
        "start_price",
        "avg_entry_price",
        "last_trade_price",
        "is_active",
        "entry_bucket_0_price",
        "entry_bucket_0_qty",
        "entry_bucket_0_amount",
        "entry_bucket_0_side",
        "entry_bucket_0_status",
        "entry_bucket_0_is_last",
        "entry_bucket_0_is_market",
        "entry_bucket_1_price",
        "entry_bucket_1_qty",
        "entry_bucket_1_amount",
        "entry_bucket_1_side",
        "entry_bucket_1_status",
        "entry_bucket_1_is_last",
        "entry_bucket_1_is_market",
        "entry_bucket_2_price",
        "entry_bucket_2_qty",
        "entry_bucket_2_amount",
        "entry_bucket_2_side",
        "entry_bucket_2_status",
        "entry_bucket_2_is_last",
        "entry_bucket_2_is_market",
        "entry_bucket_3_price",
        "entry_bucket_3_qty",
        "entry_bucket_3_amount",
        "entry_bucket_3_side",
        "entry_bucket_3_status",
        "entry_bucket_3_is_last",
        "entry_bucket_3_is_market",
        "exit_bucket_0_price",
        "exit_bucket_0_qty",
        "exit_bucket_0_amount",
        "exit_bucket_0_side",
        "exit_bucket_0_status",
        "exit_bucket_0_is_last",
        "exit_bucket_0_is_market",
        "exit_bucket_1_price",
        "exit_bucket_1_qty",
        "exit_bucket_1_amount",
        "exit_bucket_1_side",
        "exit_bucket_1_status",
        "exit_bucket_1_is_last",
        "exit_bucket_1_is_market",
        "exit_bucket_2_price",
        "exit_bucket_2_qty",
        "exit_bucket_2_amount",
        "exit_bucket_2_side",
        "exit_bucket_2_status",
        "exit_bucket_2_is_last",
        "exit_bucket_2_is_market",
        "exit_bucket_3_price",
        "exit_bucket_3_qty",
        "exit_bucket_3_amount",
        "exit_bucket_3_side",
        "exit_bucket_3_status",
        "exit_bucket_3_is_last",
        "exit_bucket_3_is_market",
        "entry_0_executor_id",
        "entry_0_executor_status",
        "entry_0_executor_is_active_and_not_filled",
        "entry_0_executor_is_full_filled",
        "entry_0_executor_is_terminated",
        "entry_0_executor_avg_entry_price",
        "entry_0_executor_qty",
        "entry_0_executor_amount",
        "entry_1_executor_id",
        "entry_1_executor_status",
        "entry_1_executor_is_active_and_not_filled",
        "entry_1_executor_is_full_filled",
        "entry_1_executor_is_terminated",
        "entry_1_executor_avg_entry_price",
        "entry_1_executor_qty",
        "entry_1_executor_amount",
        "entry_2_executor_id",
        "entry_2_executor_status",
        "entry_2_executor_is_active_and_not_filled",
        "entry_2_executor_is_full_filled",
        "entry_2_executor_is_terminated",
        "entry_2_executor_avg_entry_price",
        "entry_2_executor_qty",
        "entry_2_executor_amount",
        "entry_3_executor_id",
        "entry_3_executor_status",
        "entry_3_executor_is_active_and_not_filled",
        "entry_3_executor_is_full_filled",
        "entry_3_executor_is_terminated",
        "entry_3_executor_avg_entry_price",
        "entry_3_executor_qty",
        "entry_3_executor_amount",
        "exit_0_executor_id",
        "exit_0_executor_status",
        "exit_0_executor_is_active_and_not_filled",
        "exit_0_executor_is_full_filled",
        "exit_0_executor_is_terminated",
        "exit_0_executor_avg_entry_price",
        "exit_0_executor_qty",
        "exit_0_executor_amount",
        "exit_1_executor_id",
        "exit_1_executor_status",
        "exit_1_executor_is_active_and_not_filled",
        "exit_1_executor_is_full_filled",
        "exit_1_executor_is_terminated",
        "exit_1_executor_avg_entry_price",
        "exit_1_executor_qty",
        "exit_1_executor_amount",
        "exit_2_executor_id",
        "exit_2_executor_status",
        "exit_2_executor_is_active_and_not_filled",
        "exit_2_executor_is_full_filled",
        "exit_2_executor_is_terminated",
        "exit_2_executor_avg_entry_price",
        "exit_2_executor_qty",
        "exit_2_executor_amount",
        "exit_3_executor_id",
        "exit_3_executor_status" "exit_3_executor_is_active_and_not_filled",
        "exit_3_executor_is_full_filled",
        "exit_3_executor_is_terminated",
        "exit_3_executor_avg_entry_price",
        "exit_3_executor_qty",
        "exit_3_executor_amount",
    ]

    row = {
        "timestamp": time.time(),
        "cycle_num": controller.cycle_num,
        "current_qty": controller.current_qty,
        "start_price": controller.start_price,
        "avg_entry_price": controller.avg_entry_price,
        "last_trade_price": controller.last_trade_price,
        "is_active": controller.is_active,
    }

    executors_states = controller.executors_states

    for i in range(4):
        bucket_id = f"{controller.cycle_num}_entry_{i}"
        if not controller.entry_buckets.get(f"{controller.cycle_num}_entry_{i}"):
            row[f"entry_bucket_{i}_price"] = None
            row[f"entry_bucket_{i}_qty"] = None
            row[f"entry_bucket_{i}_amount"] = None
            row[f"entry_bucket_{i}_side"] = None
            row[f"entry_bucket_{i}_status"] = None
            row[f"entry_bucket_{i}_is_last"] = None
            row[f"entry_bucket_{i}_is_market"] = None
        else:
            row[f"entry_bucket_{i}_price"] = controller.entry_buckets.get(bucket_id).price
            row[f"entry_bucket_{i}_qty"] = controller.entry_buckets.get(bucket_id).qty
            row[f"entry_bucket_{i}_amount"] = controller.entry_buckets.get(bucket_id).amount
            row[f"entry_bucket_{i}_side"] = controller.entry_buckets.get(bucket_id).side
            row[f"entry_bucket_{i}_status"] = controller.entry_buckets.get(bucket_id).status
            row[f"entry_bucket_{i}_is_last"] = controller.entry_buckets.get(bucket_id).is_last
            row[f"entry_bucket_{i}_is_market"] = controller.entry_buckets.get(bucket_id).is_market

        if not executors_states.get(bucket_id):
            row[f"entry_{i}_executor_id"] = None
            row[f"entry_{i}_executor_status"] = None
            row[f"entry_{i}_executor_is_active_and_not_filled"] = None
            row[f"entry_{i}_executor_is_full_filled"] = None
            row[f"entry_{i}_executor_is_terminated"] = None
            row[f"entry_{i}_executor_avg_entry_price"] = None
            row[f"entry_{i}_executor_qty"] = None
            row[f"entry_{i}_executor_amount"] = None
        else:
            row[f"entry_{i}_executor_id"] = executors_states.get(bucket_id).id
            row[f"entry_{i}_executor_status"] = executors_states.get(bucket_id).status
            row[f"entry_{i}_executor_is_active_and_not_filled"] = executors_states.get(
                bucket_id
            ).is_active_and_not_filled
            row[f"entry_{i}_executor_is_full_filled"] = executors_states.get(bucket_id).is_full_filled
            row[f"entry_{i}_executor_is_terminated"] = executors_states.get(bucket_id).is_terminated
            row[f"entry_{i}_executor_avg_entry_price"] = executors_states.get(bucket_id).average_entry_price
            row[f"entry_{i}_executor_qty"] = executors_states.get(bucket_id).qty
            row[f"entry_{i}_executor_amount"] = executors_states.get(bucket_id).amount

        bucket_id = f"{controller.cycle_num}_exit_{i}"
        if not controller.exit_buckets.get(f"{controller.cycle_num}_exit_{i}"):
            row[f"exit_bucket_{i}_price"] = None
            row[f"exit_bucket_{i}_qty"] = None
            row[f"exit_bucket_{i}_amount"] = None
            row[f"exit_bucket_{i}_side"] = None
            row[f"exit_bucket_{i}_status"] = None
            row[f"exit_bucket_{i}_is_last"] = None
            row[f"exit_bucket_{i}_is_market"] = None
        else:
            row[f"exit_bucket_{i}_price"] = controller.exit_buckets.get(f"{controller.cycle_num}_exit_{i}").price
            row[f"exit_bucket_{i}_qty"] = controller.exit_buckets.get(f"{controller.cycle_num}_exit_{i}").qty
            row[f"exit_bucket_{i}_amount"] = controller.exit_buckets.get(f"{controller.cycle_num}_exit_{i}").amount
            row[f"exit_bucket_{i}_side"] = controller.exit_buckets.get(f"{controller.cycle_num}_exit_{i}").side
            row[f"exit_bucket_{i}_status"] = controller.exit_buckets.get(f"{controller.cycle_num}_exit_{i}").status
            row[f"exit_bucket_{i}_is_last"] = controller.exit_buckets.get(f"{controller.cycle_num}_exit_{i}").is_last
            row[f"exit_bucket_{i}_is_market"] = controller.exit_buckets.get(
                f"{controller.cycle_num}_exit_{i}"
            ).is_market

        if not executors_states.get(bucket_id):
            row[f"exit_{i}_executor_id"] = None
            row[f"exit_{i}_executor_status"] = None
            row[f"exit_{i}_executor_is_active_and_not_filled"] = None
            row[f"exit_{i}_executor_is_full_filled"] = None
            row[f"exit_{i}_executor_is_terminated"] = None
            row[f"exit_{i}_executor_avg_entry_price"] = None
            row[f"exit_{i}_executor_qty"] = None
            row[f"exit_{i}_executor_amount"] = None
        else:
            row[f"exit_{i}_executor_id"] = executors_states.get(bucket_id).id
            row[f"exit_{i}_executor_status"] = executors_states.get(bucket_id).status
            row[f"exit_{i}_executor_is_active_and_not_filled"] = executors_states.get(
                bucket_id
            ).is_active_and_not_filled
            row[f"exit_{i}_executor_is_full_filled"] = executors_states.get(bucket_id).is_full_filled
            row[f"exit_{i}_executor_is_terminated"] = executors_states.get(bucket_id).is_terminated
            row[f"exit_{i}_executor_avg_entry_price"] = executors_states.get(bucket_id).average_entry_price
            row[f"exit_{i}_executor_qty"] = executors_states.get(bucket_id).qty
            row[f"exit_{i}_executor_amount"] = executors_states.get(bucket_id).amount

    if not os.path.exists(filename):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(row.values())
    else:
        with open(filename, "a") as f:
            writer = csv.writer(f)
            writer.writerow(row.values())
