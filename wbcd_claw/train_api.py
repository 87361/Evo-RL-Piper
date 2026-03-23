"""Training task CRUD API routes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from wbcd_claw.train_manager import TrainManager

router = APIRouter(prefix="/api/train", tags=["training"])

_manager: TrainManager | None = None


class LaunchPayload(BaseModel):
    config_file: str
    exp_name: str = ""


def init_train_state(manager: TrainManager) -> None:
    global _manager
    _manager = manager


@router.get("/configs")
def list_configs():
    return {"ok": True, "configs": _manager.list_configs()}


@router.post("/launch")
def launch(payload: LaunchPayload):
    result = _manager.launch(payload.config_file, payload.exp_name)
    return result


@router.get("/tasks")
def list_tasks(limit: int = 30):
    tasks = _manager.list_tasks(limit)
    return {"ok": True, "tasks": tasks}


@router.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = _manager.get_task(task_id)
    if not task:
        return {"ok": False, "error": "task not found"}
    return {"ok": True, "task": task}


@router.get("/tasks/{task_id}/logs")
def get_logs(task_id: str, lines: int = 80):
    return _manager.get_logs(task_id, lines)


@router.post("/tasks/{task_id}/kill")
def kill_task(task_id: str):
    return _manager.kill_task(task_id)
