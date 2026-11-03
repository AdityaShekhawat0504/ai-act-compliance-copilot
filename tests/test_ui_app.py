"""Smoke test for the Streamlit UI. Skipped when streamlit is not installed;
CI runs it with `uv run --with streamlit`."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_renders_without_errors() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception


def test_demo_generates_pack() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.checkbox[0].check().run()
    at.button[0].click().run()
    assert not at.exception
    assert any("High-risk" in str(m.value) for m in at.metric)
