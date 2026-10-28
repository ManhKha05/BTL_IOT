"""
Root runner script for Assistive AI Server.
Delegates to the modular FastAPI application in app.main.
"""
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows to support Vietnamese characters in console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure the server directory is on sys.path
SERVER_ROOT = Path(__file__).resolve().parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from app.core.config import settings
from app.main import app

if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on {settings.HOST}:{settings.PORT} ...")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
