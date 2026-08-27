from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def test_app_loads_expected_sections() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not app.exception
    labels = [tab.label for tab in app.tabs]
    for expected in (
        "Overview",
        "Method comparison",
        "Convergence",
        "Damping",
        "Resonance & beats",
        "Nonlinear pendulum",
        "External data",
        "Validation",
    ):
        assert expected in labels
    assert any(metric.label == "Exact reference" for metric in app.metric)


def test_method_comparison_and_compliance_run() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    _button(app, "Run method comparison").click()
    app.run(timeout=30)
    assert not app.exception
    _button(app, "Run compliance suite").click()
    app.run(timeout=30)
    assert not app.exception
