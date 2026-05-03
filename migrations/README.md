# Mnemosyne migrations

Alembic-managed schema migrations for the `memory` schema.

## Driver split

The runtime stack uses `asyncpg` (configured in `src/mnemosyne/db/`).
Alembic ships with sync-only drivers, so `alembic upgrade` requires
`psycopg2`. We install `psycopg2-binary` in the `dev` extras group so
that contributors and CI can run migrations without polluting the
runtime dependency set.

## Running migrations locally

```bash
pip install -e '.[dev]'
export DATABASE_URL=postgresql://user:pass@localhost:5432/mnemosyne
alembic upgrade head
```

## CI / production

Production uses the same flow: install the `dev` extras in the migration
job only, run `alembic upgrade head`, and tear the job down. The
application container does **not** ship `psycopg2`; runtime queries go
through `asyncpg`.

## Why not an async Alembic env

An async-aware `env.py` is feasible (see Alembic's
`run_async_migrations` recipe) but adds maintenance burden for a
release-time tool. We chose the sync-driver split for simplicity.

## Raw-SQL fallback

If `alembic upgrade` is unavailable in a given environment (for
example, the migration job has not yet adopted the dev extras),
applying the migration files as raw SQL via `psql` is supported. Each
versioned migration in `migrations/versions/` should remain runnable as
plain DDL.
