# from decimal import Decimal
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


"""
Вход в позиции осуществляется по DCA.

"""


class EntryBucket(BaseModel):
    id: str
    drawdown_percentage: float # На сколько процентов должна упасть цена от средней для входа.
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
    level_id: str
    is_active: bool
    is_filled: bool
    average_entry_price: float
    amount: float
    qty: float


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
    @property
    def state(self) -> Dict[str, Any]:
        return {}

    @property
    def executors_states(self) -> Dict[str, Any]:
        executors_states = {
            ex.custom_info["level_id"]: ExecutorState(
                id=ex.id,
                level_id=ex.custom_info["level_id"],
                is_active=ex.is_active and not ex.is_trading,
                is_filled=ex.is_active and ex.is_trading,
                average_entry_price=ex.custom_info["current_position_average_price"],
                amount=ex.filled_amount_quote,
                qty=ex.filled_amount_quote / ex.custom_info["current_position_average_price"],
            )
            for ex in self.executors_info
        }
        return executors_states

    def __init__(self, config: BucketsV2ControllerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.config.entry_buckets = self.sort_config_entry_buckets(self.config.entry_buckets, self.config.entry_side)
        self.config.exit_buckets = self.sort_config_exit_buckets(self.config.exit_buckets, self.config.entry_side)
        self.position_mode = self.config.position_mode
        self.entry_side = self.config.entry_side
        self.exit_side = TradeType.SELL if self.entry_side == TradeType.BUY else TradeType.BUY
        self.entry_buckets: Dict[str, GridBucket] = {}
        self.exit_buckets: Dict[str, GridBucket] = {}
        self._entry_grid_updated_at = 0
        self._exit_grid_updated_at = 0
        self.current_qty: float = 0.0
        self.start_price: float = self.get_last_trade_price()
        self.last_trade_price = self.start_price
        self.avg_entry_price: float = self.start_price
        self.executors_actions: List[ExecutorAction] = []
        self.trading_rules = None
        self.entry_buckets: Dict[str, GridBucket] = {}
        self.exit_buckets: Dict[str, GridBucket] = {}
        self.trades: List[Any] = []
        self.is_active = True
        self.is_last_exit_bucket_filled = False
        self._is_new_bucket_filled = False
        self._is_new_entry_bucket_filled = False

    def sort_config_entry_buckets(self, buckets: list[EntryBucket], trade_side: TradeType) -> list[EntryBucket]:
        return buckets

    def sort_config_exit_buckets(self, buckets: list[ExitBucket], trade_side: TradeType) -> list[ExitBucket]:
        return sorted(buckets, key=lambda x: x.profit_percentage)


    def _calculate_entry_buckets(self) -> Dict[str, GridBucket]:
        self.logger().info("calculate_entry_buckets")
        self.logger().info("Entry buckets: %s", self.entry_buckets)
        self.trading_rules = self.market_data_provider.get_trading_rules(
            self.config.connector_name, self.config.trading_pair
        )
        buckets = {}
        min_amount = self.config.min_order_amount

        for bucket in self.config.entry_buckets:
            if self.entry_buckets.get(bucket.id) and self.entry_buckets[bucket.id].status == GridBucketStatus.FILLED:
                buckets[bucket.id] = self.entry_buckets[bucket.id]
                continue

            if self.entry_side == TradeType.BUY:
                price = self.avg_entry_price * (1 - bucket.drawdown_percentage / 100)
            else:
                price = self.avg_entry_price * (1 + bucket.drawdown_percentage / 100)

            price_quantized = self.market_data_provider.quantize_order_price(
                self.config.connector_name, self.config.trading_pair, price
            )
            qty = bucket.amount * self.config.leverage / price_quantized
            qty_quantized = self.market_data_provider.quantize_order_amount(
                self.config.connector_name, self.config.trading_pair, qty
            )
            amount = qty_quantized * price_quantized
            status = GridBucketStatus.NOT_ACTIVE if amount >= min_amount else GridBucketStatus.SKIPPED

            buckets[bucket.id] = GridBucket(
                id=bucket.id,
                price=price_quantized,
                qty=qty_quantized,
                amount=qty_quantized * price_quantized,
                side=self.entry_side,
                status=status,
                is_market=False if bucket.drawdown_percentage != 0 else True
            )
            self.logger().info("BUCKET: %s", buckets[bucket.id])
        return buckets

    def _calculate_exit_buckets(self) -> Dict[str, GridBucket]:
        self.logger().info(f"calculate_exit_buckets. Exit side: {self.exit_side}")
        buckets = {}
        self.logger().info("current_qty: %s", self.current_qty)
        if self.current_qty == 0:
            return buckets
        min_amount = self.config.min_order_amount
        available_qty = self.current_qty    
        average_entry_price = self.avg_entry_price

        for bucket in self.config.exit_buckets:
            if self.exit_buckets.get(bucket.id) and self.exit_buckets[bucket.id].status == GridBucketStatus.FILLED:
                buckets[bucket.id] = self.exit_buckets[bucket.id]
                continue

            if self.exit_side == TradeType.BUY:
                bucket_price = average_entry_price * (1 - bucket.profit_percentage / 100)
            else:
                bucket_price = average_entry_price * (1 + bucket.profit_percentage / 100)
            price_quantized = self.market_data_provider.quantize_order_price(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                price=bucket_price,
            )

            qty = bucket.qty_share * available_qty
            qty_quantized = self.market_data_provider.quantize_order_amount(
                self.config.connector_name, self.config.trading_pair, qty
            )
            amount = qty_quantized * price_quantized
            if amount >= min_amount:
                status = GridBucketStatus.NOT_ACTIVE
                available_qty -= qty_quantized
            else:
                status = GridBucketStatus.SKIPPED

            buckets[bucket.id] = GridBucket(
                id=bucket.id,
                price=price_quantized,
                qty=qty_quantized,
                amount=amount,
                side=self.exit_side,
                status=status,
            )

            self.logger().debug("available_qty: %s", available_qty)
            self.logger().info("buckets: %s", buckets)
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
            if not executors_states.get(bucket.id):
                result_buckets[bucket_id] = bucket
                continue
            if bucket.status == GridBucketStatus.FILLED:
                result_buckets[bucket_id] = bucket
                continue

            if executors_states[bucket_id].is_active:
                bucket.status = GridBucketStatus.ACTIVE
            
            elif executors_states[bucket_id].is_filled:
                self._is_new_bucket_filled = True
                calculated_qty = self.current_qty + (
                    executors_states[bucket_id].qty
                    if bucket.side == TradeType.BUY
                    else -executors_states[bucket_id].qty
                )
                # Get the base symbol from the trading pair (e.g., 'BTC' from 'BTC-USDT')
                base_symbol = self.config.trading_pair.split('-')[0]
                exchange_qty = self.market_data_provider.get_balance(self.config.connector_name, base_symbol)  
                self.current_qty = min(calculated_qty, exchange_qty) 

                bucket.status = GridBucketStatus.FILLED
                bucket.price = executors_states[bucket_id].average_entry_price
                bucket.qty = executors_states[bucket_id].qty
                bucket.amount = executors_states[bucket_id].amount
                if bucket.is_last:
                    self.is_last_exit_bucket_filled = True

            result_buckets[bucket_id] = bucket
        self.logger().info(f"handle executors updates. ENTRY_BUCKETS: {self.entry_buckets}\n grid_buckets: {grid_buckets}\n result_buckets: {result_buckets}")

        return result_buckets


    def get_current_qty_from_exchange(self) -> float:
        # Get the base symbol from the trading pair (e.g., 'BTC' from 'BTC-USDT')
        base_symbol = self.config.trading_pair.split('-')[0]
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
        self.logger().info("Last exit bucket filled")
        stop_actions = [
            StopExecutorAction(controller_id=self.config.id, executor_id=executor.id, keep_position=True)
            for executor in self.executors_info
        ]
        self.executors_actions.extend(stop_actions)
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
        self.last_trade_price = self.market_data_provider.quantize_order_price(
            self.config.connector_name,
            self.config.trading_pair,
            price=self.market_data_provider.get_price_by_type(
                self.config.connector_name,
                self.config.trading_pair,
                PriceType.LastTrade,
            ),
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
            self.logger().info(f"AVERAGE_ENTRY_PRICE: {price}")
            return price
        self.logger().info(f"AVERAGE_ENTRY_PRICE is START_PRICE: {self.start_price}")
        return self.start_price


    # def calculate_returns(self) -> Decimal:
    #     """Calculate the current returns based on position and entry price.

    #     Returns:
    #         Decimal: The current returns as a percentage. Positive for profit, negative for loss.
    #     """
    #     if self.current_qty == Decimal("0") or self.avg_entry_price == Decimal("0"):
    #         return Decimal("0")

    #     current_price = self.get_last_trade_price()
    #     if current_price == Decimal("0"):
    #         return Decimal("0")

    #     if self.entry_side == TradeType.BUY:
    #         return ((current_price - self.avg_entry_price) / self.avg_entry_price) * Decimal("100")
    #     else:  # SHORT position
    #         return ((self.avg_entry_price - current_price) / self.avg_entry_price) * Decimal("100")


    async def update_processed_data(self):
        """
        Update the processed data based on the current state of the strategy.
        """
        if self.start_price.is_nan():
            self.logger().info("START_PRICE is NaN. Fetching last trade price...")
            self.start_price = self.get_last_trade_price()
            if self.avg_entry_price.is_nan():
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
        
        avg_entry_price = self.avg_entry_price
        self.entry_buckets = self._handle_executors_updates(self.entry_buckets)
        self.logger().info(f"ENTRY BUCKETS after handle executors: {self.entry_buckets}")
        self._is_new_entry_bucket_filled = self._is_new_bucket_filled
        self._is_new_bucket_filled = False

        if self._is_new_entry_bucket_filled:
            self.avg_entry_price = self.calculate_average_long_entry_price()

        self.exit_buckets = self._handle_executors_updates(self.exit_buckets)
        self._is_new_bucket_filled = False


        if avg_entry_price != self.avg_entry_price:
            self.logger().info(f"AVERAGE entry price updated to {self.avg_entry_price}")
        
        if self._is_new_entry_bucket_filled:
            self.entry_buckets = self._calculate_entry_buckets()
            self._determine_entry_executors()
            self.exit_buckets = self._calculate_exit_buckets()
            self._determine_exit_executors()
            self._is_new_entry_bucket_filled = False

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

    def _determine_entry_executors(self):
        actions = []
        for bucket_id, bucket in self.entry_buckets.items():

            if bucket.status == GridBucketStatus.ACTIVE and self.executors_states[bucket_id].is_active:
                continue

            keep_position = bucket.status == GridBucketStatus.FILLED
            # stop current bucket executor
            if self.executors_states.get(bucket_id):
                self.logger().info(f"STOP ENTRY EXECUTOR. {bucket}")
                self.logger().info(f"EXECUTOR STATE: {self.executors_states[bucket_id]}")
                actions.append(
                    StopExecutorAction(
                        controller_id=self.config.id,
                        executor_id=self.executors_states[bucket_id].id,
                        keep_position=keep_position,
                    )
                )
                
            if bucket.status == GridBucketStatus.FILLED:
                continue
            if bucket.status == GridBucketStatus.SKIPPED:
                continue

            # create new bucket executor
            self.logger().info(f"NEW ENTRY EXECUTOR. {bucket}")
            actions.append(
                CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=self.get_limit_entry_executor_config(
                        price=bucket.price,
                        qty=bucket.qty,
                        level_id=bucket.id,
                    ),
                )
            )
            # place only one entry order per once
            break

        return actions

    def _determine_exit_executors(self):
        actions = []
        for bucket_id, bucket in self.exit_buckets.items():

            keep_position = bucket.status == GridBucketStatus.FILLED

            # stop current bucket executor
            if self.executors_states.get(bucket_id):
                actions.append(
                    StopExecutorAction(
                        controller_id=self.config.id,
                        executor_id=self.executors_states[bucket_id].id,
                        keep_position=keep_position,
                    )
                )
                
            if bucket.status == GridBucketStatus.FILLED:
                continue
            if bucket.status == GridBucketStatus.SKIPPED:
                continue

            if bucket.side == TradeType.BUY:
                price = min(self.last_trade_price, bucket.price)
            
            else:
                price = max(self.last_trade_price, bucket.price)

            # create new bucket executor
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

        return actions

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        Determine actions based on the provided executor handler report.
        """
        # self.logger().info(f"Determin executor actions. start_price: {self.start_price}")
        if self.start_price.is_nan():
            return []
        if self.entry_buckets == {}:
            self.entry_buckets = self._calculate_entry_buckets()
        actions = self.executors_actions
        # actions.extend(self.determine_create_executor_actions())
        # actions.extend(self.determine_stop_executor_actions())
        actions.extend(self._determine_entry_executors())
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
        self.is_last_exit_bucket_filled = False
        self._calculate_entry_buckets()


    def on_stop(self):
        stop_actions = [
            StopExecutorAction(controller_id=self.config.id, executor_id=executor.id, keep_position=True)
            for executor in self.executors_info
        ]
        self.executors_actions.extend(stop_actions)
        self.is_active = False
        self.logger().info("ON STOP")

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
