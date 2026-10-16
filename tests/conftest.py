import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS_FILE = ".pytest_results.jsonl"

_RESULTS_PATH = None
_IS_WORKER = False


def _is_xdist_worker(config):
    return getattr(config, "workerinput", None) is not None


def pytest_configure(config):
    global _RESULTS_PATH, _IS_WORKER
    _RESULTS_PATH = Path(str(config.rootdir)) / RESULTS_FILE
    _IS_WORKER = _is_xdist_worker(config)
    if _IS_WORKER:
        return
    _RESULTS_PATH.write_text("")


def pytest_runtest_logreport(report):
    if _IS_WORKER or _RESULTS_PATH is None:
        return
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "nodeid": report.nodeid,
        "phase": getattr(report, "when", "collect"),
        "outcome": report.outcome,
        "duration": round(
            getattr(report, "duration", 0.0) or 0.0, 4
        ),
    }
    if report.longrepr is not None:
        entry["longrepr"] = str(report.longrepr)
    with _RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def pytest_sessionfinish(session):
    if _IS_WORKER or _RESULTS_PATH is None:
        return
    with _RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(
                        timespec="milliseconds"
                    ),
                    "event": "session finished",
                    "failed": session.testsfailed,
                    "collected": session.testscollected,
                }
            )
            + "\n"
        )


def pytest_addoption(parser):
    parser.addoption(
        "--env",
        action="store",
        default="dev",
        help="Environment to run tests against",
    )
