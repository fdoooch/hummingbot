from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.models.base import RunnableStatus
from decimal import Decimal

p = ExecutorInfo(
    id='Bj7t95f98c5zwXqktEBuLrqKdLSZVoaRsvZqqYdjqPwT', 
    timestamp=1742308255.0199704, 
    type='position_executor', 
    close_timestamp=1742308258.0, 
    close_type=CloseType.POSITION_HOLD, 
    status=RunnableStatus.TERMINATED, 
    config=PositionExecutorConfig(
        id='Bj7t95f98c5zwXqktEBuLrqKdLSZVoaRsvZqqYdjqPwT', 
        type='position_executor', 
        timestamp=1742308255.0199704, 
        controller_id='FPzm95DBUhwyiGDzKhQHucRaCUKphjwyh2RKVg4BPrQy', 
        trading_pair='LTC-USDT', 
        connector_name='okx', 
        side=TradeType.BUY, 
        entry_price=Decimal('88.26'), 
        amount=Decimal('0.135961'), 
        triple_barrier_config=TripleBarrierConfig(
            stop_loss=None, 
            take_profit=None, 
            time_limit=None, 
            trailing_stop=None, 
            open_order_type=OrderType.MARKET, 
            take_profit_order_type=OrderType.MARKET, 
            stop_loss_order_type=OrderType.MARKET, 
            time_limit_order_type=OrderType.MARKET
        ), 
        leverage=1, 
        activation_bounds=[Decimal('0.1'), Decimal('0.1')], 
        level_id='0_entry_0'
    ), 
    net_pnl_pct=Decimal('-0.003839573117398123823122507921'), 
    net_pnl_quote=Decimal('-0.04591891348800000000000000000'), 
    cum_fees_quote=Decimal('0.043209493488'), 
    filled_amount_quote=Decimal('11.95937988'), 
    is_active=False, 
    is_trading=False, 
    custom_info={
        'level_id': '0_entry_0', 
        'current_position_average_price': Decimal('88.28'), 
        'side': TradeType.BUY, 
        'current_retries': 0, 
        'max_retries': 10, 
        'close_price': Decimal('88.26'), 
        'order_ids': ['93027a12dac34fBCBLCUT59d76ee0224'], 
        'held_position_orders': [
            {
                'client_order_id': '93027a12dac34fBCBLCUT59d76ee0224', 
                'exchange_order_id': '2342295875458539520', 
                'trading_pair': 'LTC-USDT', 
                'order_type': 'MARKET', 
                'trade_type': 'BUY', 
                'price': '88.26', 
                'amount': '0.135961', 
                'executed_amount_base': '0.135961', 
                'executed_amount_quote': '12.00263708', 
                'last_state': '5', 
                'leverage': '1', 
                'position': 'NIL', 
                'creation_timestamp': 1742308255.0, 
                'last_update_timestamp': 1742308256.808, 
                'order_fills': {
                    '153860908': {
                        'trade_id': '153860908', 
                        'client_order_id': '93027a12dac34fBCBLCUT59d76ee0224', 
                        'exchange_order_id': '2342295875458539520', 
                        'trading_pair': 'LTC-USDT', 
                        'fill_timestamp': 1742308256.808, 
                        'fill_price': '88.28', 
                        'fill_base_amount': '0.135961', 
                        'fill_quote_amount': '12.00263708', 
                        'fee': {
                            'fee_type': 'AddedToCost', 
                            'percent': '0', 
                            'percent_token': 'LTC', 
                            'flat_fees': [
                                {'token': 'LTC', 'amount': '0.0004894596'}
                            ]
                        }, 
                        'is_taker': True
                    }
                }, 
                'cumulative_fee_paid_base': 0.0004894596, 
                'cumulative_fee_paid_quote': 0.043209493488
            }
        ]
    }, 
    controller_id='FPzm95DBUhwyiGDzKhQHucRaCUKphjwyh2RKVg4BPrQy'
)