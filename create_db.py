import asyncio
from db import engine, Base
from models import User, Instrument, OrderBook, Transaction, Balance, Order

# alembic revision --autogenerate -m "added Instruments"
# alembic upgrade head

async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

asyncio.run(init_models())