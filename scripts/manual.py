"""
Manual mode UI/API for the OpenEvolve visualizer

This module exposes:
  - GET  /manual                    (manual tasks page)
  - GET  /manual/api/tasks          (list tasks, pending only)
  - GET  /manual/api/tasks/<id>
  - POST /manual/api/tasks/<id>/answer

Task queue directory is assumed to be:
  <run_root>/manual_tasks_queue

where run_root corresponds to the OpenEvolve run output directory (typically "openevolve_output")
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from flask import Blueprint, jsonify, render_template, request

QUEUE_DIRNAME = "manual_tasks_queue"


def _resolve_run_root(path_str: str) -> Path:
    """
    Resolve run root from a path that may point to:
      - <run_root>
      - <run_root>/checkpoints
      - <run_root>/checkpoints/checkpoint_123
      - <run_root>/checkpoint_123

    Returns:
      Path to <run_root>
    """
    p = Path(path_str).expanduser().resolve()

    if p.name.startswith("checkpoint_"):
        # .../checkpoints/checkpoint_123 -> run_root is parent of "checkpoints"
        if p.parent.name == "checkpoints":
            return p.parent.parent
        # .../checkpoint_123 -> run_root is parent
        return p.parent

    if p.name == "checkpoints":
        return p.parent

    return p


def _queue_dir(run_root: Path) -> Path:
    return run_root / QUEUE_DIRNAME


@dataclass
class TaskItem:
    id: str
    created_at: str
    model: Optional[str]
    has_answer: bool


def _list_tasks(qdir: Path) -> List[TaskItem]:
    if not qdir.exists():
        return []

    tasks: List[TaskItem] = []

    for p in sorted(qdir.glob("*.json")):
        # Ignore hidden and non-task JSON files
        if p.name.startswith("."):
            continue
        if p.name.endswith(".answer.json"):
            continue

        task_id = p.stem
        answer = qdir / f"{task_id}.answer.json"
        has_answer = answer.exists()

        created_at = ""
        model = None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            created_at = str(data.get("created_at") or "")
            model = data.get("model")
        except Exception:
            pass

        tasks.append(TaskItem(id=task_id, created_at=created_at, model=model, has_answer=has_answer))

    # Show only pending
    tasks = [t for t in tasks if not t.has_answer]
    return tasks


def _read_task(qdir: Path, task_id: str) -> Optional[Dict]:
    p = qdir / f"{task_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalize_newlines(text: str) -> str:
    """
    Normalize CRLF/CR newlines to LF

    This makes diff parsing deterministic because OpenEvolve diff regexes use '\n'
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip()


def _write_answer(qdir: Path, task_id: str, answer_text: str) -> None:
    qdir.mkdir(parents=True, exist_ok=True)
    out = qdir / f"{task_id}.answer.json"
    tmp = qdir / f".{task_id}.answer.json.tmp"

    answer_text = _normalize_newlines(answer_text)

    payload = {"id": task_id, "answer": answer_text}
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)


def create_manual_blueprint(get_visualizer_path: Callable[[], str]) -> Blueprint:
    bp = Blueprint("manual", __name__, url_prefix="/manual")

    @bp.route("", methods=["GET"], strict_slashes=False)
    @bp.route("/", methods=["GET"], strict_slashes=False)
    def manual_page():
        return render_template("manual_page.html")

    @bp.get("/api/tasks")
    def api_tasks():
        run_root = _resolve_run_root(get_visualizer_path())
        qdir = _queue_dir(run_root)
        items = _list_tasks(qdir)
        data = [{"id": t.id, "created_at": t.created_at, "model": t.model} for t in items]
        return jsonify({"tasks": data})

    @bp.get("/api/tasks/<task_id>")
    def api_task_detail(task_id: str):
        run_root = _resolve_run_root(get_visualizer_path())
        qdir = _queue_dir(run_root)
        data = _read_task(qdir, task_id)
        if data is None:
            return ("Task not found", 404)

        return jsonify({
            "id": data.get("id"),
            "created_at": data.get("created_at"),
            "model": data.get("model"),
            "display_prompt": data.get("display_prompt", ""),
        })

    @bp.post("/api/tasks/<task_id>/answer")
    def api_task_answer(task_id: str):
        run_root = _resolve_run_root(get_visualizer_path())
        qdir = _queue_dir(run_root)

        # Accept form POST (current UI), optionally JSON too.
        answer = (request.form.get("answer") or "").strip()
        if not answer and request.is_json:
            body = request.get_json(silent=True) or {}
            answer = str(body.get("answer") or "").strip()

        if not answer:
            return ("Answer must not be empty", 400)

        if not (qdir / f"{task_id}.json").exists():
            return ("Task not found", 404)

        _write_answer(qdir, task_id, answer)
        return jsonify({"ok": True})

    return bp
