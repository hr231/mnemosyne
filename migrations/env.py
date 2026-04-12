import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv

load_dotenv()  # pick up MNEMOSYNE_PG_DSN from .env at repo root

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Allow MNEMOSYNE_PG_DSN env var to override whatever alembic.ini says.
_dsn = os.environ.get("MNEMOSYNE_PG_DSN") or config.get_main_option("sqlalchemy.url")


def run_migrations_offline():
    context.configure(url=_dsn, target_metadata=None, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    from sqlalchemy import create_engine
    connectable = create_engine(_dsn)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
