from fastapi import FastAPI, HTTPException, Header, status, Security, Response, Request, Query
from fastapi.params import Depends
from pydantic import BaseModel, Field, field_validator
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from typing import Optional, Dict, List, Union
import uuid
from sqlalchemy.exc import IntegrityError
from uuid import UUID, uuid4
from models import User, Instrument, OrderBook, Transaction, Balance, Order
from sqlalchemy import select, asc, desc, and_, delete, func, literal
from sqlalchemy.ext.asyncio import AsyncSession
from db import get_db, AsyncSessionLocal
from enum import Enum
from datetime import datetime
import httpx
import requests
from fastapi.openapi.utils import get_openapi

#uvicorn main:app --reload --host 0.0.0.0 --port 8000

#python -m uvicorn main:app --reload

METADATA_URL = "http://169.254.169.254/computeMetadata/v1/instance/id"
METADATA_HEADERS = {"Metadata-Flavor": "Google"}

tags_metadata = [
    {
        "name": "Public",
        "description": "Публичные операции",
    },
    {
        "name": "Balance",
        "description": "Операции с балансом",
    },
    {
        "name": "Order",
        "description": "Управление заказами",
    },
{
        "name": "Admin",
        "description": "Админское управление",
    },
{
        "name": "User",
        "description": "Управление пользователями",
    },
{
        "name": "Cloud Functions",
        "description": "Cloud Functions",
    },
]

app = FastAPI(openapi_tags=tags_metadata)

api_key_header = APIKeyHeader(
    name="Authorization",
    description="TOKEN <api_key>",
    auto_error=True
)

async def get_or_create_balance(
    db: AsyncSession,
    user_id: UUID,
    instrument_id: UUID
) -> Balance:
    balance = await db.execute(
        select(Balance)
        .where(and_(
            Balance.user_id == user_id,
            Balance.instrument_id == instrument_id
        )))
    balance = balance.scalar_one_or_none()
    if not balance:
        balance = Balance(
            user_id=user_id,
            instrument_id=instrument_id,
            amount=0,
            reserved=0
        )
        db.add(balance)
    return balance

class UserRole(str, Enum):
    USER = "USER"
    ADMIN = "ADMIN"

class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class Status(str, Enum):
    NEW = "NEW"
    EXECUTED = "EXECUTED"
    PARTIALLY_EXECUTED = "PARTIALLY_EXECUTED"
    CANCELLED = "CANCELLED"

def generate_api_key() -> str:
    return f"key-{uuid4()}"

class RegisterRequest(BaseModel):
    name: str

class RegisterResponse(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    role: UserRole
    api_key: str = Field(..., example="key-123e4567-e89b-12d3-a456-426614174000")

class OrderItem(BaseModel):
    price: int = Field(..., ge=1)
    qty: int = Field(..., gt=0)

class L2Order(BaseModel):
    bid_levels: List[OrderItem]
    ask_levels: List[OrderItem]

class InstrumentItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    ticker: str = Field(pattern=r'^[A-Z]{2,10}$')
    name: str

class TransactionItem(BaseModel):
    ticker: str
    amount: int = Field(..., gt=0)
    price: int = Field(..., ge=1)
    timestamp: datetime = Field(..., example="2025-05-27T14:30:00Z")

class BalanceDepositResponse(BaseModel):
    success: bool

class BalanceDepositRequest(BaseModel):
    user_id: UUID = Field(default_factory=uuid4)
    ticker: str
    amount: int = Field(..., gt=0)

class InstrumentAddRequest(BaseModel):
    name: str
    ticker: str = Field(..., min_length=2, max_length=10, pattern=r"^[A-Z]+$")

    @field_validator('ticker')
    def uppercase_ticker(cls, v):
        return v.upper()

class InstrumentAddResponse(BaseModel):
    success: bool

class InstrumentDeleteResponse(BaseModel):
    success: bool

class OrderBody(BaseModel):
    direction: Direction
    ticker: str
    qty: int = Field(..., ge=1)
    price: Optional[int] = Field(None, gt=0)

class OrderCreateRequest(BaseModel):
    direction: Direction
    ticker: str
    qty: int = Field(..., ge=1)
    price: Optional[int] = Field(None, gt=0)  # Теперь необязательное поле

class OrderCreateResponse(BaseModel):
    success: bool
    order_id: str

class OrderGetResponse(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    status: Status
    user_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(..., example="2025-05-27T14:30:00Z")
    body: OrderBody
    filled: int

class OrderDeleteResponse(BaseModel):
    success: bool

def validate_uuid(uuid_str: str):
    try:
        return UUID(uuid_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid ID format"
        )

@app.on_event("startup")
async def startup_event():
    async with AsyncSessionLocal() as db:  # Используем контекстный менеджер сессии
        result = await db.execute(select(Instrument).where(Instrument.ticker == "RUB"))
        if not result.scalar_one_or_none():
            db.add(Instrument(ticker="RUB", name="Russian Ruble"))
            await db.commit()

# Public
@app.post("/api/v1/public/register",
          response_model=RegisterResponse,
          status_code=status.HTTP_200_OK,
          summary="Регистрация пользователя в платформе",
          tags=["Public"])
async def register_user(
        payload: RegisterRequest,
        db: AsyncSession = Depends(get_db)):
    api_key = f"key-{uuid.uuid4()}"
    user = User(
        name = payload.name,
        api_key = api_key)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

@app.get("/api/v1/public/instrument",
         tags=["Public"],
         response_model=List[InstrumentItem],
         summary="Список доступных инструментов")
async def get_instruments(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Instrument))
    instruments = result.scalars().all()
    return instruments

@app.get("/api/v1/public/orderbook/{ticker}",
        response_model=L2Order,
        tags=["Public"],
         summary="Получить стакан заявок по тикеру")
async def get_orderbook_by_ticker(
        ticker: str,
        limit: int = Query(10, ge=1, le=50),
        db: AsyncSession = Depends(get_db),):
    result_bid = await db.execute(
        select(OrderBook.price, func.sum(OrderBook.qty).label("total_qty"))
        .where(and_(OrderBook.ticker == ticker, OrderBook.side == "bid"))
        .group_by(OrderBook.price)
        .order_by(desc(OrderBook.price))
        .limit(limit)
    )
    bid_levels = [OrderItem(price=price, qty=total_qty) for price, total_qty in result_bid]

    result_ask = await db.execute(
        select(OrderBook.price, func.sum(OrderBook.qty).label("total_qty"))
        .where(and_(OrderBook.ticker == ticker, OrderBook.side == "ask"))
        .group_by(OrderBook.price)
        .order_by(asc(OrderBook.price))
        .limit(limit)
    )
    ask_levels = [OrderItem(price=price, qty=total_qty) for price, total_qty in result_ask]

    return L2Order(bid_levels=bid_levels, ask_levels=ask_levels)

@app.get("/api/v1/public/transactions/{ticker}",
         tags=["Public"],
         response_model=List[TransactionItem],
         summary="Получить транзакции по тикеру")
async def get_transactions_by_ticker(
        ticker: str,
        limit: int = Query(10, ge=1, le=50),
        db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Transaction)
        .where(Transaction.ticker == ticker)
        .order_by(desc(Transaction.timestamp))
        .limit(limit))
    transactions = [
        TransactionItem(
            ticker=row.ticker,
            amount=row.qty,
            price=row.price,
            timestamp=row.timestamp)
        for row in result.scalars()
    ]
    # 3) просто возвращаем список (он может быть пустым)
    return transactions

@app.post("/api/v1/order",
          response_model=OrderCreateResponse,
          tags=["Order"],
          summary="Создать заказ")
async def create_order(
        data: OrderCreateRequest,
        authorization: str = Security(api_key_header),
        db: AsyncSession = Depends(get_db)):
    if not authorization.upper().startswith("TOKEN "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme")
    token = authorization.split(" ", 1)[1]
    result = await db.execute(
        select(User).where(User.api_key == token))
    requesting_user = result.scalars().first()
    if not requesting_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key")

    # Проверка что пользователь существует
    user_exists = await db.scalar(
        select(1).where(User.id == requesting_user.id))
    if not user_exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User does not exist")

    # Проверка доступности инструмента
    instrument = await db.execute(
        select(Instrument).where(Instrument.ticker == data.ticker)
    )
    instrument = instrument.scalar_one_or_none()
    if not instrument:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Instrument with ticker {data.ticker} not found")

    # Проверяем существование RUB
    rub_instrument = await db.execute(
        select(Instrument).where(Instrument.ticker == "RUB")
    )
    rub_instrument = rub_instrument.scalar_one_or_none()
    if not rub_instrument:
        # Создаем RUB если не существует
        rub_instrument = Instrument(ticker="RUB", name="Russian Ruble")
        db.add(rub_instrument)
        await db.flush()
        # 4) Создаём сам ордер (NEW, filled=0)
        order = Order(
            user_id=requesting_user.id,
            direction=data.direction,
            ticker=data.ticker,
            qty=data.qty,
            price=data.price,
            status=Status.NEW,
            timestamp=datetime.utcnow(),
            filled=0
        )
        db.add(order)
        await db.flush()

        # 5) Для лимитных — сразу в стакан
        if data.price is not None:
            book_entry = OrderBook(
                order_id=order.id,
                ticker=data.ticker,
                side="bid" if data.direction == Direction.BUY else "ask",
                price=data.price,
                qty=data.qty
            )
            db.add(book_entry)
            await db.flush()

        # 6) Отправляем в процессинг (там и резерв, и matching)
        try:
            if data.price is None:
                await process_market_order(db, order, instrument)
            else:
                await process_limit_order(db, order, instrument)
        except HTTPException:
            # пробрасываем HTTPException как есть
            raise
        except Exception as e:
            # в случае непредвиденной ошибки откатываем статус
            order.status = Status.CANCELLED
            db.add(order)
            await db.flush()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing order: {str(e)}"
            )

        # 7) Финальный коммит и ответ
        await db.commit()
        return OrderCreateResponse(success=True, order_id=str(order.id))

async def get_max_ask_price(db: AsyncSession, ticker: str) -> int:
    result = await db.execute(
        select(func.max(OrderBook.price))
        .where(and_(
            OrderBook.ticker == ticker,
            OrderBook.side == "ask"
        ))
    )
    return result.scalar() or 0

async def process_market_order(db: AsyncSession, order: Order, instrument: Instrument):
    rub_instrument = await db.execute(
        select(Instrument).where(Instrument.ticker == "RUB")
    )
    rub_instrument = rub_instrument.scalar_one_or_none()

    if not rub_instrument:
        rub_instrument = Instrument(ticker="RUB", name="Russian Ruble")
        db.add(rub_instrument)
        await db.flush()

    token_balance = await get_or_create_balance(db, order.user_id, instrument.id)
    rub_balance = await get_or_create_balance(db, order.user_id, rub_instrument.id)

    remaining = order.qty
    initial_reserved = 0

    # Определяем направление ордера
    if order.direction == Direction.BUY:
        q = select(Order).where(and_(
            Order.ticker == order.ticker,
            Order.direction == Direction.SELL,
            Order.status.in_([Status.NEW, Status.PARTIALLY_EXECUTED]),
            Order.price.isnot(None)
        )).order_by(asc(Order.price), asc(Order.timestamp))
    else:  # SELL
        q = select(Order).where(and_(
            Order.ticker == order.ticker,
            Order.direction == Direction.BUY,
            Order.status.in_([Status.NEW, Status.PARTIALLY_EXECUTED]),
            Order.price.isnot(None)
        )).order_by(desc(Order.price), asc(Order.timestamp))

    matching = (await db.execute(q)).scalars().all()
    # считаем initial_reserved
    for mo in matching:
        avail = mo.qty - mo.filled
        take = min(remaining, avail)
        initial_reserved += take * mo.price
        remaining -= take
        if remaining == 0:
            break

    if remaining > 0:
        # недостаточно ликвидности
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Insufficient liquidity for market buy"
                if order.direction == Direction.BUY
                else "No bids available for market sell"
            )
        )

    # 3) Резервируем средства или инструмент
    if order.direction == Direction.BUY:
        rub_balance.reserved += initial_reserved
        db.add(rub_balance)
    else:  # SELL
        token_balance.reserved += order.qty
        db.add(token_balance)

    await db.flush()

    # 4) Исполняем сами сделки
    remaining_qty = order.qty
    actual_cost = 0

    for match_order in matching:
        if remaining_qty == 0:
            break

        avail = match_order.qty - match_order.filled
        exec_qty = min(remaining_qty, avail)
        price = match_order.price
        cost = exec_qty * price

        # подрезаем последний шаг, если не хватает резерва
        if order.direction == Direction.BUY and cost > initial_reserved - actual_cost:
            max_qty = (initial_reserved - actual_cost) // price
            if max_qty == 0:
                break
            exec_qty = max_qty
            cost = exec_qty * price

        await execute_trade(
            db=db,
            ticker=order.ticker,
            qty=exec_qty,
            price=price,
            match_order=match_order,
            buyer_id=(order.user_id if order.direction == Direction.BUY else match_order.user_id),
            seller_id=(match_order.user_id if order.direction == Direction.BUY else order.user_id),
            instrument_id=instrument.id
        )

        remaining_qty -= exec_qty
        actual_cost += cost

        # обновляем исходный order
        order.filled += exec_qty
        order.status = (
            Status.EXECUTED
            if order.filled >= order.qty
            else Status.PARTIALLY_EXECUTED
        )
        db.add(order)

    # 5) Возвращаем неиспользованный резерв
    if order.direction == Direction.BUY:
        unused = initial_reserved - actual_cost
        if unused > 0:
            rub_balance.reserved = max(0, rub_balance.reserved - unused)
            db.add(rub_balance)
    else:
        # если остались непроданные токены — снимаем резерв токена
        unsold = remaining_qty
        if unsold > 0:
            token_balance.reserved = max(0, token_balance.reserved - unsold)
            db.add(token_balance)

    await db.flush()


async def process_limit_order(db: AsyncSession, order: Order, instrument: Instrument):
    # Сначала пытаемся найти немедленное совпадение
    if order.direction == Direction.BUY:
        # Для покупки: ищем предложения на продажу по цене <= нашей
        matching_orders = await db.execute(
            select(Order)
            .where(and_(
                Order.ticker == order.ticker,
                Order.direction == Direction.SELL,
                Order.status.in_([Status.NEW, Status.PARTIALLY_EXECUTED]),
                Order.price <= order.price
            ))
            .order_by(asc(Order.price), asc(Order.timestamp))
        )
    else:  # SELL
        # Для продажи: ищем предложения на покупку по цене >= нашей
        matching_orders = await db.execute(
            select(Order)
            .where(and_(
                Order.ticker == order.ticker,
                Order.direction == Direction.BUY,
                Order.status.in_([Status.NEW, Status.PARTIALLY_EXECUTED]),
                Order.price >= order.price
            ))
            .order_by(desc(Order.price))
        )

    matching_orders = matching_orders.scalars().all()
    remaining_qty = order.qty

    if order.direction == Direction.BUY:
        q = select(Order).where(and_(
            Order.ticker == order.ticker,
            Order.direction == Direction.SELL,
            Order.status.in_([Status.NEW, Status.PARTIALLY_EXECUTED]),
            Order.price <= order.price)).order_by(asc(Order.price), asc(Order.timestamp))
    else:
        q = select(Order).where(and_(
            Order.ticker == order.ticker,
            Order.direction == Direction.BUY,
            Order.status.in_([Status.NEW, Status.PARTIALLY_EXECUTED]),
            Order.price >= order.price
            )).order_by(desc(Order.price), asc(Order.timestamp))

    matching_orders = (await db.execute(q)).scalars().all()
    remaining_qty = order.qty

    # Исполняем совпадающие ордера
    for match_order in matching_orders:
        if remaining_qty <= 0:
            break

        available_qty = match_order.qty - match_order.filled
        execution_qty = min(remaining_qty, available_qty)
        execution_price = match_order.price

        # Создаем транзакцию
        await execute_trade(
            db=db,
            ticker=order.ticker,
            qty=execution_qty,
            match_order=match_order,
            price=execution_price,
            buyer_id=order.user_id if order.direction == Direction.BUY else match_order.user_id,
            seller_id=order.user_id if order.direction == Direction.SELL else match_order.user_id,
            instrument_id=instrument.id
        )

        # Обновляем статус и заполненное количество
        order.filled += execution_qty
        match_order.filled += execution_qty

        # Обновляем статусы ордеров
        order.status = Status.EXECUTED if order.filled >= order.qty else Status.PARTIALLY_EXECUTED
        match_order.status = Status.EXECUTED if match_order.filled >= match_order.qty else Status.PARTIALLY_EXECUTED

        db.add(order)
        db.add(match_order)

        remaining_qty -= execution_qty

        if order.status == Status.EXECUTED:
            break

    if order.filled == 0:
        order.status = Status.NEW
    elif 0 < order.filled < order.qty:
        order.status = Status.PARTIALLY_EXECUTED
    else:
        order.status = Status.EXECUTED
    # Если после исполнения осталась неисполненная часть, добавляем в стакан
    if remaining_qty > 0 and order.status != Status.EXECUTED:
        # Ищем существующую запись в стакане
        existing_book = await db.execute(
            select(OrderBook)
            .where(OrderBook.order_id == order.id)
        )
        existing_book = existing_book.scalar_one_or_none()

        if existing_book:
            # Обновляем существующую запись
            existing_book.qty = remaining_qty
            db.add(existing_book)
        else:
            # Создаем новую запись
            order_book = OrderBook(
                order_id=order.id,
                ticker=order.ticker,
                side="bid" if order.direction == Direction.BUY else "ask",
                price=order.price,
                qty=remaining_qty
            )
            db.add(order_book)
    elif order.status == Status.EXECUTED:
        # Удаляем запись из стакана при полном исполнении
        await db.execute(
            delete(OrderBook)
            .where(OrderBook.order_id == order.id)
        )

    await db.flush()


async def execute_trade(
    db: AsyncSession,
    ticker: str,
    qty: int,
    price: int,
    match_order: Order,
    buyer_id: UUID,
    seller_id: UUID,
    instrument_id: UUID
):
    # Получаем RUB инструмент
    rub_instrument = await db.execute(
        select(Instrument).where(Instrument.ticker == "RUB")
    )
    rub_instrument = rub_instrument.scalar_one_or_none()
    if not rub_instrument:
        rub_instrument = Instrument(ticker="RUB", name="Russian Ruble")
        db.add(rub_instrument)
        await db.flush()

    # Рассчитываем общую стоимость сделки
    total_cost = price * qty

    # Обновляем баланс покупателя (актив)
    buyer_balance = await get_or_create_balance(db, buyer_id, instrument_id)
    buyer_balance.amount += qty
    db.add(buyer_balance)

    # Обновляем баланс продавца (актив)
    seller_balance = await get_or_create_balance(db, seller_id, instrument_id)
    seller_balance.amount -= qty
    # Уменьшаем резерв продавца
    seller_balance.reserved = max(0, seller_balance.reserved - qty)
    db.add(seller_balance)

    # Обновляем RUB баланс покупателя (списание)
    buyer_rub_balance = await get_or_create_balance(db, buyer_id, rub_instrument.id)
    buyer_rub_balance.amount -= total_cost
    # Уменьшаем резерв покупателя
    buyer_rub_balance.reserved = max(0, buyer_rub_balance.reserved - total_cost)
    db.add(buyer_rub_balance)

    # Обновляем RUB баланс продавца (зачисление)
    seller_rub_balance = await get_or_create_balance(db, seller_id, rub_instrument.id)
    seller_rub_balance.amount += total_cost
    db.add(seller_rub_balance)

    # Создаем транзакции (отдельно для покупателя и продавца)
    buyer_transaction = Transaction(
        ticker=ticker,
        qty=qty,
        price=price,
        side='buy',
        user_id=buyer_id
    )
    seller_transaction = Transaction(
        ticker=ticker,
        qty=qty,
        price=price,
        side='sell',
        user_id=seller_id
    )
    db.add_all([buyer_transaction, seller_transaction])

    db.add_all(
        [buyer_balance, seller_balance, buyer_rub_balance, seller_rub_balance, buyer_transaction, seller_transaction])
    await db.flush()

    match_order.filled += qty
    match_order.status = Status.EXECUTED if match_order.filled >= match_order.qty else Status.PARTIALLY_EXECUTED
    db.add(match_order)

    # скорректировать запись в OrderBook для встречного ордера
    # (match_order.id совпадает с order_id в OrderBook)
    res = await db.execute(
        select(OrderBook)
        .where(OrderBook.order_id == match_order.id)
    )
    ob = await db.get(OrderBook, match_order.id)

    if ob is not None:
        remaining = match_order.qty - match_order.filled
        if remaining > 0:
            ob.qty = remaining
            db.add(ob)
    else:
        await db.delete(ob)

    await db.flush()


@app.get("/api/v1/order",
          tags=["Order"],
         response_model=List[OrderGetResponse],
          summary="Список заказов")
async def list_orders(
    authorization: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db)):
    if not authorization.upper().startswith("TOKEN "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme")
    token = authorization.split(" ", 1)[1]
    result = await db.execute(
        select(User).where(User.api_key == token))
    requesting_user = result.scalars().first()
    if not requesting_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key")
    result = await db.execute(
        select(Order))
    orders = result.scalars().all()
    return [OrderGetResponse(
        id=o.id,
        status=o.status,
        user_id=o.user_id,
        timestamp=o.timestamp,
        body=OrderBody(
            direction=o.direction,
            ticker=o.ticker,
            qty=o.qty,
            price=o.price
        ),
        filled=o.filled
    ) for o in orders]

@app.get("/api/v1/order/{order_id}",
          tags=["Order"],
          response_model=OrderGetResponse,
          summary="Получить заказ")
async def get_order(
    order_id: str,
    authorization: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db)):
    order_uuid = validate_uuid(order_id)
    if not authorization.upper().startswith("TOKEN "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme")
    token = authorization.split(" ", 1)[1]
    result = await db.execute(
        select(User).where(User.api_key == token))
    requesting_user = result.scalars().first()
    if not requesting_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key")
    result = await db.execute(
        select(Order)
        .where(Order.id == order_uuid))
    order = result.scalar_one_or_none()

    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order with this order id not found")

    # Используем текущие данные ордера
    return OrderGetResponse(
        id=order.id,
        status=order.status,
        user_id=order.user_id,
        timestamp=order.timestamp,
        body=OrderBody(
            direction=order.direction,
            ticker=order.ticker,
            qty=order.qty,
            price=order.price,  # Может быть None для рыночных ордеров
        ),
        filled=order.filled)

@app.delete("/api/v1/order/{order_id}",
          response_model=OrderDeleteResponse,
          tags=["Order"],
          summary="Отменить заказ")
async def cancel_order(
    order_id: str,
    authorization: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db)):
    order_uuid = validate_uuid(order_id)
    if not authorization.upper().startswith("TOKEN "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme")
    token = authorization.split(" ", 1)[1]
    result = await db.execute(
        select(User).where(User.api_key == token))
    requesting_user = result.scalars().first()
    if not requesting_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key")
    # Получаем ордер
    result = await db.execute(
        select(Order)
        .where(Order.id == order_uuid))
    order = result.scalar_one_or_none()

    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found")

    # Проверка прав доступа: только владелец или админ
    if order.user_id != requesting_user.id and requesting_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot cancel another user's order")

    # Проверка статуса: нельзя отменить исполненный или уже отмененный ордер
    if order.status in [Status.EXECUTED, Status.CANCELLED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel order in {order.status} status")

    # 4) Если рыночный ордер, просто помечаем Cancelled
    if order.price is None:
        order.status = Status.CANCELLED
        db.add(order)

    else:
        # Это лимитный ордер — возвращаем резерв и подправляем стакан
        unfilled = order.qty - order.filled

        # 4a) Снимаем резерв
        if order.direction == Direction.SELL:
            # резервы токена
            bal_q = await db.execute(
                select(Balance)
                .where(and_(
                    Balance.user_id == order.user_id,
                    Balance.instrument_id == await db.scalar(
                        select(Instrument.id).where(Instrument.ticker == order.ticker)
                    )
                ))
            )
            bal = bal_q.scalar_one_or_none()
            if bal:
                bal.reserved = max(0, bal.reserved - unfilled)
                db.add(bal)

        else:  # limit BUY
            rub_id = await db.scalar(
                select(Instrument.id).where(Instrument.ticker == "RUB")
            )
            rub_bal = await get_or_create_balance(db, order.user_id, rub_id)
            rub_bal.reserved = max(0, rub_bal.reserved - unfilled * order.price)
            db.add(rub_bal)

        # 4b) Обновляем или удаляем запись из OrderBook
        result = await db.execute(
            select(OrderBook)
            .where(OrderBook.order_id == order.id)
        )
        ob = result.scalar_one_or_none()

        if ob:
            unfilled = order.qty - order.filled
            if unfilled > 0:
                # обновляем остаток
                ob.qty = unfilled
                db.add(ob)
            else:
                # удаляем запись, если ничего не осталось
                await db.delete(ob)

        order.status = Status.CANCELLED
        db.add(order)

    # 5) Сохраняем изменения
    await db.commit()
    return OrderDeleteResponse(success=True)

@app.delete("/api/v1/admin/user/{user_id}",
         tags=["User", "Admin"],
         response_model=RegisterResponse,
         summary="Удалить пользователя",)
async def delete_user_by_id(
        user_id: str,
        authorization: str = Security(api_key_header),
        db: AsyncSession = Depends(get_db)):
    user_uuid = validate_uuid(user_id)
    if not authorization.upper().startswith("TOKEN "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme")
    token = authorization.split(" ", 1)[1]
    result = await db.execute(
        select(User).where(User.api_key == token))
    requesting_user = result.scalars().first()
    if not requesting_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key")
    if (requesting_user.role != UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can delete users")
    #if user_id == requesting_user.id:
    #    raise HTTPException(
    #        status_code=403,
    #        detail="You cannot delete yourself"
    #    )
    result = await db.execute(
        select(User)
        .where(User.id == user_uuid))
    user_to_delete = result.scalar_one_or_none()
    if not user_to_delete:
        raise HTTPException(status_code=404, detail="User not found")
    #if requesting_user.role != "ADMIN":
    #    raise HTTPException(
    #        status_code=403,
    #        detail="Users can't delete admin profiles"
    #    )
    await db.delete(user_to_delete)
    await db.commit()
    return RegisterResponse(
        id=user_to_delete.id,
        name=user_to_delete.name,
        role=user_to_delete.role,
        api_key=user_to_delete.api_key)

@app.post("/api/v1/admin/instrument",
          tags=["Admin"],
          response_model=InstrumentAddResponse,
          summary="Создать инструмент")
async def create_instrument(
    data: InstrumentAddRequest,
    authorization: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db)):
    if not data.name.strip():##########################
        raise HTTPException(400, "Name must be non‑empty") #################
    if not authorization.upper().startswith("TOKEN "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme")
    token = authorization.split(" ", 1)[1]
    result = await db.execute(
        select(User).where(User.api_key == token))
    requesting_user = result.scalars().first()
    if not requesting_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key")
    if (requesting_user.role != UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can create Instruments")
    existing = await db.execute(select(Instrument).where(Instrument.ticker == data.ticker))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail = f"Instrument with ticker {data.ticker} already exists")
    #if existing.scalar_one_or_none():
    #    return InstrumentAddResponse(success=True)
    instrument = Instrument(
        ticker=data.ticker,
        name=data.name)
    db.add(instrument)
    await db.commit()
    await db.refresh(instrument)
    return InstrumentAddResponse(success=True)

@app.delete("/api/v1/admin/instrument/{ticker}",
          tags=["Admin"],
          response_model=InstrumentDeleteResponse,
          summary="Удалить инструмент")
async def delete_instrument_by_ticker(
    ticker: str,
    authorization: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db)):
    if not authorization.upper().startswith("TOKEN "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme")
    token = authorization.split(" ", 1)[1]
    result = await db.execute(
        select(User).where(User.api_key == token))
    requesting_user = result.scalars().first()
    if not requesting_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key")
    if (requesting_user.role != UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can delete Instruments")
    result = await db.execute(
        select(Instrument)
        .where(Instrument.ticker == ticker))
    instrument = result.scalar_one_or_none()
    if not instrument:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instrument with {ticker} ticker not found")
    try:
        await db.execute(delete(Balance).where(Balance.instrument_id == instrument.id))
        await db.execute(delete(Order).where(Order.ticker == ticker))
        await db.execute(delete(OrderBook).where(OrderBook.ticker == ticker))
        await db.execute(delete(Transaction).where(Transaction.ticker == ticker))
        await db.delete(instrument)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete instrument: there are existing balances or orders referencing it"
        )
    return InstrumentDeleteResponse(success=True)

@app.get("/api/v1/balance", tags=["Balance"],
         response_model=Dict[str,float],
         responses={
             200: {
                 "description": "Баланс пользователя",
                 "content": {
                     "application/json": {
                         "example": {
                             "MEMCOIN": 0,
                             "DODGE": 100500
                         }
                     }
                 }
             }
         },
         summary="Получить баланс пользователя")
async def get_balances(
        authorization: str = Security(api_key_header),
        db: AsyncSession = Depends(get_db)):
    if not authorization.upper().startswith("TOKEN "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme")
    token = authorization.split(" ", 1)[1]
    result = await db.execute(
        select(User).where(User.api_key == token))
    requesting_user = result.scalars().first()
    if not requesting_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key")

    stmt = select(
        Instrument.ticker,
        func.coalesce(Balance.amount, 0).label("amount")
    ).select_from(Instrument
                  ).outerjoin(
        Balance,
        and_(
            Balance.user_id == requesting_user.id,
            Balance.instrument_id == Instrument.id
        )
    ).order_by(Instrument.ticker)

    # Добавляем RUB, если его нет в инструментах
    rub_exists = await db.execute(
        select(1).where(Instrument.ticker == "RUB")
    )
    if not rub_exists.scalar():
        # Добавляем RUB в результат, если его нет в БД
        stmt = stmt.union_all(
            select(
                literal("RUB").label("ticker"),
                literal(0).label("amount")
            )
        )

    balances_result = await db.execute(stmt)
    balances = balances_result.all()

    # Форматируем результат в словарь
    return {ticker: amount for ticker, amount in balances}

@app.post("/api/v1/admin/balance/deposit",
          response_model=BalanceDepositResponse,
          tags=["Balance", "Admin"],
          summary="Пополнить счёт пользователя")
async def deposit(
    data: BalanceDepositRequest,
    authorization: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db)):
    if not authorization.upper().startswith("TOKEN "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme")
    token = authorization.split(" ", 1)[1]
    result = await db.execute(
        select(User).where(User.api_key == token))
    requesting_user = result.scalars().first()
    if not requesting_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key")

    # Проверка прав администратора
    if requesting_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can deposit funds")

    # Поиск инструмента по тикеру
    instrument = await db.execute(
        select(Instrument).where(Instrument.ticker == data.ticker))
    instrument = instrument.scalar_one_or_none()

    if not instrument:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instrument with ticker {data.ticker} not found")

    # Получение или создание баланса
    balance = await get_or_create_balance(db, data.user_id, instrument.id)

    # Пополнение баланса
    balance.amount += data.amount
    db.add(balance)
    await db.commit()

    return BalanceDepositResponse(success=True)

@app.post("/api/v1/admin/balance/withdraw",
          response_model=BalanceDepositResponse,
          tags=["Balance", "Admin"],
          summary="Вывести с счета пользователя")
async def withdraw(
    data: BalanceDepositRequest,
    authorization: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db)):
    if not authorization.upper().startswith("TOKEN "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme")
    token = authorization.split(" ", 1)[1]
    result = await db.execute(
        select(User).where(User.api_key == token))
    requesting_user = result.scalars().first()
    if not requesting_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key")
    # Проверка прав администратора
    if requesting_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can withdraw funds")

    # Поиск инструмента по тикеру
    instrument = await db.execute(
        select(Instrument).where(Instrument.ticker == data.ticker))
    instrument = instrument.scalar_one_or_none()

    if not instrument:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instrument with ticker {data.ticker} not found")

    # Получение баланса
    balance = await db.execute(
        select(Balance)
        .where(and_(
            Balance.user_id == data.user_id,
            Balance.instrument_id == instrument.id
        )))
    balance = balance.scalar_one_or_none()

    if not balance:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has no balance for this instrument")

    # Проверка достаточности средств
    available = balance.amount - balance.reserved
    if available < data.amount:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insufficient available balance. Available: {available}, Requested: {data.amount}")

    # Снятие средств
    balance.amount -= data.amount
    db.add(balance)
    await db.commit()

    return BalanceDepositResponse(success=True)
