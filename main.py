from fastapi import FastAPI, HTTPException, Header, status, Security, Response, Request, Query
from fastapi.params import Depends
from pydantic import BaseModel, Field, field_validator
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from typing import Optional, Dict, List, Union
import uuid
from uuid import UUID, uuid4
from models import User, Instrument, OrderBook, Transaction, Balance, Order
from sqlalchemy import select, asc, desc, and_, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from db import get_db
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
    price: int = Field(..., gt=0)

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
        select(OrderBook)
        .where(and_(OrderBook.ticker == ticker, OrderBook.side == "bid"))
        .order_by(desc(OrderBook.price))
        .limit(limit))
    bid_levels = [OrderItem(price=row.price, qty=row.qty) for row in result_bid.scalars()]
    result_ask = await db.execute(
        select(OrderBook)
        .where(and_(OrderBook.ticker == ticker, OrderBook.side == "ask"))
        .order_by(asc(OrderBook.price))
        .limit(limit))
    ask_levels = [OrderItem(price=row.price, qty=row.qty) for row in result_ask.scalars()]
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
    transactions = [TransactionItem(ticker=row.ticker, amount=row.qty, price=row.price, timestamp=row.timestamp)
                    for row in result.scalars()]
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
        # Проверка доступности инструмента
    instrument = await db.execute(
        select(Instrument).where(Instrument.ticker == data.ticker)
    )
    instrument = instrument.scalar_one_or_none()
    if not instrument:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Instrument with ticker {data.ticker} not found")

    # Для ордеров на продажу: проверка и резервирование баланса
    if data.direction == Direction.SELL:
        # Получаем баланс пользователя
        balance = await db.execute(
            select(Balance)
            .where(and_(
                Balance.user_id == requesting_user.id,
                Balance.instrument_id == instrument.id
            ))
        )
        balance = balance.scalar_one_or_none()

        # Рассчитываем доступное количество
        available = balance.amount if balance else 0

        # Вычитаем уже зарезервированное в других ордерах
        reserved_result = await db.execute(
            select(func.sum(Order.qty - Order.filled))
            .where(and_(
                Order.user_id == requesting_user.id,
                Order.ticker == data.ticker,
                Order.direction == Direction.SELL,
                Order.status.in_([Status.NEW, Status.PARTIALLY_EXECUTED])
            ))
        )
        reserved = reserved_result.scalar() or 0
        available -= reserved

        if available < data.qty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient available balance. Available: {available}, Requested: {data.qty}"
            )

    # Создаем ордер
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
    await db.commit()
    await db.refresh(order)

    # Обработка ордера
    try:
        if data.price is None:  # Рыночный ордер
            await process_market_order(db, order, instrument)
        else:  # Лимитный ордер
            await process_limit_order(db, order, instrument)
    except Exception as e:
        # Отменяем ордер в случае ошибки
        order.status = Status.CANCELLED
        db.add(order)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing order: {str(e)}"
        )

    return OrderCreateResponse(
        success=True,
        order_id=str(order.id))


async def process_market_order(db: AsyncSession, order: Order, instrument: Instrument):
    # Определяем направление ордера
    if order.direction == Direction.BUY:
        # Для покупки: ищем лучшие предложения на продажу (самые низкие цены)
        best_orders = await db.execute(
            select(Order)
            .where(and_(
                Order.ticker == order.ticker,
                Order.direction == Direction.SELL,
                Order.status.in_([Status.NEW, Status.PARTIALLY_EXECUTED]),
                Order.price.isnot(None)
            ))
            .order_by(asc(Order.price))
        )
    else:  # SELL
        # Для продажи: ищем лучшие предложения на покупку (самые высокие цены)
        best_orders = await db.execute(
            select(Order)
            .where(and_(
                Order.ticker == order.ticker,
                Order.direction == Direction.BUY,
                Order.status.in_([Status.NEW, Status.PARTIALLY_EXECUTED]),
                Order.price.isnot(None)
            ))
            .order_by(desc(Order.price))
        )

    matching_orders = best_orders.scalars().all()
    remaining_qty = order.qty

    # Исполняем ордер по мере нахождения совпадений
    for match_order in matching_orders:
        if remaining_qty <= 0:
            break

        # Доступное количество в совпадающем ордере
        available_qty = match_order.qty - match_order.filled
        execution_qty = min(remaining_qty, available_qty)
        execution_price = match_order.price

        # Создаем транзакцию
        await execute_trade(
            db=db,
            ticker=order.ticker,
            qty=execution_qty,
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

        # Если ордер полностью исполнен, выходим
        if order.status == Status.EXECUTED:
            break

    await db.commit()


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
            .order_by(asc(Order.price))
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

    # Если после исполнения осталась неисполненная часть, добавляем в стакан
    if remaining_qty > 0 and order.status != Status.EXECUTED:
        # Добавляем в таблицу стакана с ссылкой на ордер
        order_book = OrderBook(
            order_id=order.id,  # Добавляем ссылку на ордер
            ticker=order.ticker,
            side="BID" if order.direction == Direction.BUY else "ASK",
            price=order.price,
            qty=remaining_qty
        )
        db.add(order_book)

    await db.commit()


async def execute_trade(
    db: AsyncSession,
    ticker: str,
    qty: int,
    price: int,
    buyer_id: UUID,
    seller_id: UUID,
    instrument_id: UUID
):
    # Обновляем баланс покупателя
    buyer_balance = await db.execute(
        select(Balance)
        .where(and_(
            Balance.user_id == buyer_id,
            Balance.instrument_id == instrument_id
        ))
    )
    buyer_balance = buyer_balance.scalar_one_or_none()
    if not buyer_balance:
        buyer_balance = Balance(
            user_id=buyer_id,
            instrument_id=instrument_id,
            amount=qty
        )
        db.add(buyer_balance)
    else:
        buyer_balance.amount += qty

    # Обновляем баланс продавца
    seller_balance = await db.execute(
        select(Balance)
        .where(and_(
            Balance.user_id == seller_id,
            Balance.instrument_id == instrument_id
        ))
    )
    seller_balance = seller_balance.scalar_one_or_none()

    if not seller_balance:
        # В реальной системе это должно быть ошибкой
        seller_balance = Balance(
            user_id=seller_id,
            instrument_id=instrument_id,
            amount=-qty,
            reserved=0
        )
        db.add(seller_balance)
    else:
        # Уменьшаем общее количество и резерв
        seller_balance.amount -= qty
        seller_balance.reserved -= qty  # Ключевое добавление!

        # Защита от отрицательных значений
        if seller_balance.reserved < 0:
            seller_balance.reserved = 0

        db.add(seller_balance)

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
    db.add(buyer_transaction)
    db.add(seller_transaction)

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
    return orders

@app.get("/api/v1/order/{order_id}",
          tags=["Order"],
          response_model=OrderGetResponse,
          summary="Получить заказ")
async def get_order(
    order_id: str,
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
        select(Order)
        .where(Order.id == order_id))
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
        .where(Order.id == order_id))
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

    # Для ордеров на продажу: освобождаем зарезервированные средства
    if order.direction == Direction.SELL:
        # Находим инструмент
        instrument = await db.execute(
            select(Instrument).where(Instrument.ticker == order.ticker)
        )
        instrument = instrument.scalar_one_or_none()

        if instrument:
            # Находим баланс пользователя
            balance = await db.execute(
                select(Balance)
                .where(and_(
                    Balance.user_id == order.user_id,
                    Balance.instrument_id == instrument.id
                ))
            )
            balance = balance.scalar_one_or_none()

            if balance:
                # Рассчитываем неисполненную часть
                unfilled_qty = order.qty - order.filled
                balance.reserved -= unfilled_qty
                if balance.reserved < 0:
                    balance.reserved = 0  # Защита от отрицательных значений
                db.add(balance)

    # Проверяем наличие в стакане перед удалением
    in_orderbook = await db.execute(
        select(OrderBook)
        .where(OrderBook.ticker == order.ticker)
        .where(OrderBook.price == order.price)
        .where(OrderBook.qty == (order.qty - order.filled))
    )
    if in_orderbook.scalar_one_or_none():
        await db.execute(
            delete(OrderBook)
            .where(OrderBook.ticker == order.ticker)
            .where(OrderBook.price == order.price)
            .where(OrderBook.qty == (order.qty - order.filled))
        )

    # Помечаем ордер как отмененный
    order.status = Status.CANCELLED
    db.add(order)
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
        .where(User.id == user_id))
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
    existing = await db.execute(select(Instrument).where(Instrument.ticker == data.ticker))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Ticker already exists")
    if (requesting_user.role != UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can create Instruments")
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
    instrument = result.scalar_one_or_none().first()
    if not instrument:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instrument with {ticker} ticker not found")
    await db.delete(instrument)
    await db.commit()
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
    # Получаем все балансы пользователя
    balances_result = await db.execute(
        select(Balance, Instrument.ticker)
        .join(Instrument, Instrument.id == Balance.instrument_id)
        .where(Balance.user_id == requesting_user.id)
    )
    balances = balances_result.all()

    # Форматируем результат
    result = {}
    for balance, ticker in balances:
        result[ticker] = balance.amount

    # Добавляем инструменты с нулевым балансом
    all_instruments = await db.execute(select(Instrument))
    for instrument in all_instruments.scalars():
        if instrument.ticker not in result:
            result[instrument.ticker] = 0

    return result

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
    result = await db.execute(
        select(Balance)
        .where(and_(Balance.user_id == data.user_id)))
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
    result = await db.execute(
        select(Balance)
        .where(and_(Balance.user_id == data.user_id)))
    return BalanceDepositResponse(success=True)

@app.get("/api/v1/hi/{name}",
         tags=["Cloud Functions"])
async def say_hi(name: str):
    # Формируем URL для вызова Yandex Cloud Function
    url = f"https://functions.yandexcloud.net/d4ebr3kc3i5nsae4jgic?name={name}"

    # Выполняем асинхронный HTTP-запрос к функции
    async with httpx.AsyncClient() as client:
        response = await client.get(url)

    # Возвращаем текстовый ответ, статус и заголовки
    return Response(
        content=response.text,
        status_code=response.status_code,
        media_type="text/plain"
    )

@app.get("/api/v1/agify/{name}",
         tags=["Cloud Functions"])
async def get_age(name: str):
    url = f"https://functions.yandexcloud.net/d4eh3018rdcg5lrq2vqu?name={name}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    return response.json()

@app.get("/instance-id",
         tags=["Cloud Functions"])
def get_instance_id():
    try:
        resp = requests.get(METADATA_URL, headers=METADATA_HEADERS, timeout=1.0)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {e}")
    return {"instance_id": resp.text}