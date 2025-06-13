import uuid

from sqlalchemy import Column, Integer, String, DateTime, func, Numeric, ForeignKey, Index
from sqlalchemy.orm import relationship
from db import Base
from sqlalchemy.dialects.postgresql import UUID

class User(Base):
    __tablename__ = "users"

    id = Column(UUID, unique=True, primary_key=True, index=True, default=uuid.uuid4)
    name = Column(String, index=True)
    role = Column(String, index=True, default="USER")
    api_key = Column(String, unique=True, index=True)

class Instrument(Base):
    __tablename__ = "instruments"

    id = Column(UUID, unique=True, primary_key=True, index=True, default=uuid.uuid4)
    ticker = Column(String, unique=True, index=True)
    name = Column(String, index=True)

class OrderBook(Base):
    __tablename__ = "order_book"

    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    order_id = Column(UUID, ForeignKey("orders.id"))  # Ссылка на ордер
    ticker = Column(String, index=True)
    side = Column(String, index=True)
    price = Column(Integer, index=True)
    qty = Column(Integer)

    __table_args__ = (
        Index('ix_order_book_ticker_side', 'ticker', 'side'),)

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)  # уникальный id транзакции
    ticker = Column(String, index=True)  # тикер инструмента
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    price = Column(Integer) # заменил с нумерик на int в таких
    qty = Column(Integer) # заменил с нумерик на int в таких
    side = Column(String)  # сторона (buy/sell)
    user_id = Column(UUID, ForeignKey("users.id"), index=True)

    user = relationship("User")

class Balance(Base):
    __tablename__ = "balances"

    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    user_id = Column(UUID, ForeignKey("users.id"), index=True)
    instrument_id = Column(UUID, ForeignKey("instruments.id"), index=True)
    amount = Column(Integer, default=0)
    reserved = Column(Integer, default=0)

    user = relationship("User")
    instrument = relationship("Instrument")

    __table_args__ = (
        Index('ix_orders_ticker_status', 'ticker', 'status'))

class Order(Base):
    __tablename__ = "orders"

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    status = Column(String, default="NEW", index=True)  # NEW, PARTIALLY_FILLED, FILLED, CANCELED
    user_id = Column(UUID, ForeignKey("users.id"), index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    direction = Column(String, index=True)  # BUY / SELL
    ticker = Column(String, index=True)
    qty = Column(Integer)
    price = Column(Integer, nullable=True, index=True)
    filled = Column(Integer, default=0)

    user = relationship("User")