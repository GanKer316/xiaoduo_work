from pathlib import Path

from streamlit.testing.v1 import AppTest

from browser_storage import hydrate_llm_config

ROOT = Path(__file__).resolve().parents[1]


def test_saved_llm_config_hydrates_even_after_empty_load_marker():
    session_state = {
        "_llm_storage_initialized": True,
        "remember_llm_config": True,
    }

    hydrated = hydrate_llm_config(
        {"api_key": "saved-key", "base_url": "https://example.test/v1", "model": "model-a"},
        session_state,
    )

    assert hydrated is True
    assert session_state["llm_api_key"] == "saved-key"
    assert session_state["llm_base_url"] == "https://example.test/v1"
    assert session_state["llm_model"] == "model-a"


def test_page_runs_default_scan_and_shows_governance_metrics():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    app.text_area[0].set_value((ROOT / "task6_business_context.md").read_text(encoding="utf-8"))
    app.text_area[1].set_value((ROOT / "task6_kb_articles.json").read_text(encoding="utf-8"))

    assert len(app.exception) == 0
    assert [button.label for button in app.button] == ["开始质量体检"]

    app.button[0].click().run()

    assert len(app.exception) == 0
    assert [metric.value for metric in app.metric] == ["40", "10", "13", "4"]
    assert list(app.dataframe[1].value.columns) == [
        "ID",
        "客户原始问题",
        "不符合业务规则的回答",
        "问题类型",
        "严重度",
    ]

def test_page_explains_when_both_inputs_are_missing():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()

    app.text_area[0].set_value("")
    app.text_area[1].set_value("")
    app.button[0].click().run()

    assert len(app.exception) == 0
    assert any("请先提供业务规则摘要和知识库 FAQ" in item.value for item in app.error)



def test_llm_mode_shows_browser_configuration_inputs():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()

    app.selectbox[0].set_value("llm").run()

    assert len(app.exception) == 0
    assert [item.label for item in app.text_input] == ["API Key", "API 地址"]

