"""The FastAPI application: creates the app, configures CORS, and wires the routes.

Run with:  uvicorn app.main:app --reload
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import ask

# Show our own INFO log lines (e.g. "no LLM call") in the uvicorn console.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")

app = FastAPI(title="NovaDesk Lite")

# CORS: lets a browser page from another origin (the future frontend,
# e.g. http://localhost:5173) call this server. Only the listed origins are allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """A simple liveness check: if this answers, the server is up."""
    return {"status": "ok"}


# Add the routes defined in routers/ask.py (POST /ask) to the app.
app.include_router(ask.router)
