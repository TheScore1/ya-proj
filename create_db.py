import asyncio
from db import engine, Base
from models import User

# alembic revision --autogenerate -m "added Instruments"
# alembic upgrade head

async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

asyncio.run(init_models())