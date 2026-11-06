"""One-command local startup, applying migrations first."""

import uvicorn
from alembic import command
from alembic.config import Config

from app.core.config import BASE_DIR

if __name__ == "__main__":
    command.upgrade(Config(str(BASE_DIR / "alembic.ini")), "head")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, access_log=False)
