from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import streamlit.components.v1 as components


_component = components.declare_component(
    "kb_quality_browser_storage",
    path=str(Path(__file__).parent / "components" / "browser_storage"),
)


def browser_storage(action: str, storage_key: str, payload: dict[str, Any] | None = None) -> Any:
    return _component(action=action, storage_key=storage_key, payload=payload or {})


def hydrate_llm_config(stored_config: Any, session_state: dict[str, Any]) -> bool:
    if not isinstance(stored_config, dict):
        return False
    if stored_config.get("status") == "empty":
        session_state["_llm_storage_initialized"] = True
        return False
    if not stored_config.get("api_key"):
        return False
    if "remember_llm_config" in session_state and not session_state["remember_llm_config"]:
        return False
    session_state["llm_api_key"] = str(stored_config.get("api_key", ""))
    session_state["llm_base_url"] = str(stored_config.get("base_url", ""))
    session_state["llm_model"] = str(stored_config.get("model", ""))
    session_state["remember_llm_config"] = True
    session_state["_llm_storage_initialized"] = True
    session_state["_saved_llm_config_fingerprint"] = hashlib.sha256(
        f"{session_state['llm_api_key']}\0{session_state['llm_base_url']}\0{session_state['llm_model']}".encode("utf-8")
    ).hexdigest()
    return True
