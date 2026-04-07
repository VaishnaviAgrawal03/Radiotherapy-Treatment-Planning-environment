"""
server.py — RadiotherapyPlanningEnv-v1 HuggingFace Space Entry Point

FastAPI server exposing:
  POST /reset
  POST /step
  GET  /state
  GET  /health

Run:
    python server.py
    or
    uvicorn server:app --host 0.0.0.0 --port 7860
"""

import os
import sys
import numpy as np
import gymnasium as gym
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional

import radiotherapy_env  # registers gym envs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="RadiotherapyPlanningEnv-v1",
    description="OpenEnv-compatible RL environment for cancer radiotherapy planning.",
    version="1.0.0",
)

# ── Global environment state ──────────────────────────────────────────────────
_env: Optional[gym.Env] = None
_last_obs: Optional[Dict] = None
_last_info: Optional[Dict] = None

DEFAULT_ENV_ID = "RadiotherapyEnv-prostate-v1"

TASK_MAP = {
    "prostate": "RadiotherapyEnv-prostate-v1",
    "head_neck": "RadiotherapyEnv-headneck-v1",
    "pediatric_brain": "RadiotherapyEnv-pediatricbrain-v1",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _numpy_to_python(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_python(v) for v in obj]
    return obj

# ── Request Models ────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task: str = "prostate"
    seed: int = 42

class StepRequest(BaseModel):
    action: int

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.post("/reset")
async def api_reset(request: Request):
    global _env, _last_obs, _last_info

    try:
        data = await request.json()
    except Exception:
        data = {}

    body = ResetRequest(
        task=data.get("task", "prostate"),
        seed=data.get("seed", 42),
    )

    env_id = TASK_MAP.get(body.task, DEFAULT_ENV_ID)

    if _env is not None:
        try:
            _env.close()
        except Exception:
            pass

    _env = gym.make(env_id)
    obs, info = _env.reset(seed=body.seed)

    _last_obs = obs
    _last_info = info

    return JSONResponse(content={
        "observation": _numpy_to_python(obs),
        "info": _numpy_to_python(info),
        "env_id": env_id,
        "task": body.task,
    })


@app.post("/step")
def api_step(body: StepRequest):
    global _env, _last_obs, _last_info

    if _env is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Call /reset first."}
        )

    if not (0 <= body.action <= 7):
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid action {body.action}. Must be 0-7."}
        )

    obs, reward, terminated, truncated, info = _env.step(body.action)

    _last_obs = obs
    _last_info = info

    return JSONResponse(content={
        "observation": _numpy_to_python(obs),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(terminated or truncated),
        "info": _numpy_to_python(info),
    })


@app.get("/state")
def api_state():
    global _env

    if _env is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Call /reset first."}
        )

    try:
        state = _env.unwrapped.state()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    return JSONResponse(content={"state": _numpy_to_python(state)})


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
