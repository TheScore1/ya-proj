import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import ssl

root_cert = os.path.expanduser("root.crt")

ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ssl_ctx.load_verify_locations(root_cert)
ssl_ctx.verify_mode = ssl.CERT_REQUIRED
ssl_ctx.check_hostname = True

DATABASE_URL = (
    "postgresql+asyncpg://appuser:2ubnCzRS@"
    "rc1a-0hq403l11fvi5ol5.mdb.yandexcloud.net:6432/db1"
)

engine = create_async_engine(DATABASE_URL, echo=True, connect_args={
        "ssl": ssl_ctx
    }
)
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

# Dependency
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
