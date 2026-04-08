"""FastAPI server exposing WellnessEnv over HTTP for HF Space deployment."""

from __future__ import annotations

import os
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse

# Add parent directory to path if running directly
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from wellness_env import WellnessEnv
from wellness_env.models import Action

app = FastAPI(title="Wellness-Outcome OpenEnv", version="1.0.0")

# One shared env instance per container — the validator expects session state
_env = WellnessEnv(seed=int(os.environ.get("SEED", "42")))


@app.post("/reset")
def reset(body: dict[str, Any] = Body({})) -> JSONResponse:
    """Start a new episode. Body: {"task_name": "single_goal"}"""
    task_name = body.get("task_name", "single_goal")
    try:
        obs = _env.reset(task_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(obs.model_dump())


@app.post("/step")
def step(body: dict[str, Any]) -> JSONResponse:
    """Apply one action. Body: Action fields as JSON."""
    try:
        action = Action(**body)
        result = _env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(result.model_dump())


@app.get("/state")
def state() -> JSONResponse:
    """Return current environment state."""
    try:
        s = _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(s.model_dump())


@app.get("/grade")
def grade() -> JSONResponse:
    """Return grader score for current episode."""
    try:
        score = _env.grade()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse({"score": score})


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


def main():
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
