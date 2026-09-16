"""
Regression test for transient TNP failures.
TNP 瞬时故障的回归测试。

TNP delegates structure prediction to NanoBodyBuilder2 (ImmuneBuilder) on the
GPU, which can fault transiently -- observed in production as
"RuntimeError: CUDA error: unspecified launch failure". TNP catches it and exits
non-zero without writing a profile. That is a property of the host at that
moment, not of the sequence: the identical sequence profiles cleanly on the next
attempt. Before the retry, a single such fault made the caller report the
molecule as undevelopable.

TNP 将结构预测委托给 GPU 上的 NanoBodyBuilder2，它可能出现瞬时故障。
这是主机当时的状态问题，而非序列问题：同一序列在下次尝试时可正常分析。

Run / 运行:
    pytest tests/test_tnp_wrapper_retry.py
    python tests/test_tnp_wrapper_retry.py
"""

import importlib.util
import json
import subprocess
from pathlib import Path

# Load the module by path: importing the metanano package pulls in bittensor,
# which this unit test does not need.
# 按路径加载模块：导入 metanano 包会引入 bittensor，本单元测试不需要它。
_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "metanano" / "utils" / "tnp_wrapper.py"
)
_spec = importlib.util.spec_from_file_location("tnp_wrapper_under_test", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)
TNPWrapper = mod.TNPWrapper

SEQUENCE = (
    "KWVLVESGGGLVKPGGSLRLSCAASGFTASRGWYGMSWFRQAPGKEREWVSLISSSGSTWMDYPDSVKG"
    "RFTSSRDNAKNSLYGQMNSLRAEDTAVYYCGVFLGAHPCPLDGEMDAFGAGTSVSVGE"
)

# A real NanoBodyBuilder2 failure, trimmed.
# 真实的 NanoBodyBuilder2 故障（已精简）。
CUDA_FAULT = (
    "Traceback (most recent call last):\n"
    '  File "/opt/ImmuneBuilder/ABodyBuilder2.py", line 211, in save\n'
    "    add_errors_as_bfactors(filename, self.error_estimates.mean(0))\n"
    "RuntimeError: CUDA error: unspecified launch failure\n"
)


def _profile_payload(seq_name):
    """A minimal TNP result JSON, shaped as bin/TNP writes it."""
    return {
        seq_name: {
            "name": seq_name,
            "Total CDR Length": 36,
            "CDR3 Length": 17,
            "CDR3 Compactness": 1.28,
            "PSH": 87.17,
            "PPC": 0.05,
            "PNC": 0.38,
            "Flags": {"L": "green", "L3": "green"},
        }
    }


def _run_with(failures, max_attempts):
    """
    Drive TNPWrapper against a TNP that fails `failures` times, then succeeds.
    让 TNPWrapper 面对一个先失败 `failures` 次、然后成功的 TNP。
    """
    state = {"calls": 0}

    def fake_run(cmd, **kwargs):
        state["calls"] += 1
        seq_name = cmd[cmd.index("--name") + 1]
        output_dir = Path(cmd[cmd.index("--output") + 1])
        if state["calls"] <= failures:
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd, output="", stderr=CUDA_FAULT
            )
        target = output_dir / f"TNP_Results_SingleSeqEntry_{seq_name}.json"
        target.write_text(json.dumps(_profile_payload(seq_name)))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    original_run, original_which = mod.subprocess.run, mod.shutil.which
    mod.subprocess.run = fake_run
    mod.shutil.which = lambda _name: "/usr/bin/TNP"
    try:
        result = TNPWrapper(max_attempts=max_attempts).profile(SEQUENCE)
        return result, state["calls"]
    finally:
        mod.subprocess.run = original_run
        mod.shutil.which = original_which


def test_transient_fault_is_retried():
    """One transient fault must not be reported as an undevelopable molecule."""
    result, calls = _run_with(failures=1, max_attempts=3)
    assert result is not None, "a transient fault must be retried, not surfaced"
    assert result.cdr3_length == 17
    assert calls == 2, f"expected one retry, saw {calls} invocation(s)"


def test_two_transient_faults_still_recover():
    """The retry budget must be usable, not just nominal."""
    result, calls = _run_with(failures=2, max_attempts=3)
    assert result is not None
    assert calls == 3


def test_persistent_failure_still_reports_none():
    """A genuinely unprofilable sequence still returns None, and does not loop."""
    result, calls = _run_with(failures=99, max_attempts=3)
    assert result is None
    assert calls == 3, f"retry budget must be bounded, saw {calls} invocation(s)"


def test_single_attempt_preserves_previous_behaviour():
    """max_attempts=1 reproduces the pre-retry contract exactly."""
    result, calls = _run_with(failures=1, max_attempts=1)
    assert result is None
    assert calls == 1


def test_stderr_is_classified_without_forwarding_raw_text():
    assert TNPWrapper._failure_categories(CUDA_FAULT) == ["cuda_launch_failure"]
    assert TNPWrapper._failure_categories("") == []
    assert TNPWrapper._failure_categories(None) == []


def _events(caplog):
    return [record.tnp_diagnostic for record in caplog.records
            if hasattr(record, "tnp_diagnostic")]


def test_exit_zero_without_profile_preserves_safe_failure_categories(monkeypatch, caplog):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        # Ordinary test markers represent untrusted text, not credentials.
        return subprocess.CompletedProcess(
            cmd, 0, "ERROR: NanoBodyBuilder2 failed to generate a model. UNTRUSTED_MARKER",
            CUDA_FAULT + " UNTRUSTED_MARKER",
        )

    monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/TNP")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert TNPWrapper().profile(SEQUENCE) is None
    assert len(calls) == 3
    failed = [event for event in _events(caplog) if event["event"] == "attempt_failed"]
    assert [event["attempt"] for event in failed] == [1, 2, 3]
    assert all(event["returncode"] == 0 for event in failed)
    assert all(set(event["categories"]) == {
        "cuda_launch_failure", "model_prediction_failed", "profile_missing",
    } for event in failed)
    assert "UNTRUSTED_MARKER" not in caplog.text
    assert SEQUENCE not in caplog.text


def test_logging_preserves_existing_per_attempt_timeout(monkeypatch, caplog):
    clock = [0.0]
    timeouts = []

    def fake_run(cmd, **kwargs):
        timeout = kwargs["timeout"]
        timeouts.append(timeout)
        clock[0] += timeout
        raise subprocess.TimeoutExpired(cmd, timeout, output="UNTRUSTED_MARKER")

    monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/TNP")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    assert TNPWrapper().profile(SEQUENCE) is None
    assert timeouts == [300.0, 300.0, 300.0]
    assert clock[0] == 900
    assert "UNTRUSTED_MARKER" not in caplog.text


def test_missing_executable_records_zero_attempts(monkeypatch, caplog):
    monkeypatch.setattr(mod.shutil, "which", lambda _: None)
    assert TNPWrapper().profile(SEQUENCE) is None
    events = _events(caplog)
    assert events[-1]["event"] == "profile_exhausted"
    assert events[-1]["attempt"] == 0
    assert events[-1]["categories"] == ["executable_missing"]


def test_parse_validation_failure_is_logged_and_retried(monkeypatch, caplog):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        name = cmd[cmd.index("--name") + 1]
        output = Path(cmd[cmd.index("--output") + 1])
        payload = _profile_payload(name)
        payload[name]["CDR3 Length"] = "not-a-number"
        (output / f"TNP_Results_SingleSeqEntry_{name}.json").write_text(json.dumps(payload))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/TNP")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert TNPWrapper().profile(SEQUENCE) is None
    assert len(calls) == 3
    failed = [event for event in _events(caplog) if event["event"] == "attempt_failed"]
    assert all(event["categories"] == ["profile_invalid"] for event in failed)


def test_concurrent_calls_have_independent_profile_ids(monkeypatch, caplog):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/TNP")
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, **_: subprocess.CompletedProcess(cmd, 0, "", ""))
    wrapper = TNPWrapper()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(wrapper.profile, [SEQUENCE, SEQUENCE]))
    assert results == [None, None]
    events = _events(caplog)
    profile_ids = {event["profile_id"] for event in events}
    assert len(profile_ids) == 2
    for profile_id in profile_ids:
        failed = [event for event in events if event["profile_id"] == profile_id
                  and event["event"] == "attempt_failed"]
        assert [event["attempt"] for event in failed] == [1, 2, 3]


def test_none_subscript_preserves_safe_traceback_locations(monkeypatch, caplog):
    trace = (
        'Traceback (most recent call last):\n'
        '  File "/scientific/bin/TNP", line 280, in main\n'
        '    UNTRUSTED_SOURCE_LINE\n'
        '  File "/untrusted/UNTRUSTED_FILENAME.py", line 999, in run\n'
        "TypeError: 'NoneType' object is not subscriptable\n"
    )

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, output="UNTRUSTED_STDOUT", stderr=trace)

    monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/TNP")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert TNPWrapper().profile(SEQUENCE) is None
    failures = [event for event in _events(caplog) if event["event"] == "attempt_failed"]
    assert [event["attempt"] for event in failures] == [1, 2, 3]
    assert all(event["categories"] == ["none_subscript"] for event in failures)
    assert all(event["traceback_locations"] == [{"component": "TNP", "line": 280}]
               for event in failures)
    assert "UNTRUSTED_" not in caplog.text


def test_traceback_locations_accept_stdout_bytes_and_bound_count():
    trace = ('  File "/scientific/bin/TNP", line 280, in main\n' * 30).encode()
    locations = TNPWrapper._traceback_locations(trace, None)
    assert len(locations) == 16
    assert all(location == {"component": "TNP", "line": 280} for location in locations)


def test_cli_null_profile_is_retried_as_compute_error(monkeypatch, caplog):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        name = cmd[cmd.index("--name") + 1]
        output = Path(cmd[cmd.index("--output") + 1])
        # TNP can return normally from an early failed step and serialize null.
        (output / f"TNP_Results_SingleSeqEntry_{name}.json").write_text(json.dumps({name: None}))
        return subprocess.CompletedProcess(cmd, 0, "UNTRUSTED_MARKER", "")

    monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/TNP")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert TNPWrapper().profile(SEQUENCE) is None
    assert len(calls) == 3
    failures = [event for event in _events(caplog) if event["event"] == "attempt_failed"]
    assert all(event["categories"] == ["profile_invalid"] for event in failures)
    assert "UNTRUSTED_MARKER" not in caplog.text


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
