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
    pytest metanano/tests/utils/test_tnp_wrapper_retry.py
    python metanano/tests/utils/test_tnp_wrapper_retry.py
"""

import importlib.util
import json
import subprocess
from pathlib import Path

# Load the module by path: importing the metanano package pulls in bittensor,
# which this unit test does not need.
# 按路径加载模块：导入 metanano 包会引入 bittensor，本单元测试不需要它。
_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "metanano" / "utils" / "tnp_wrapper.py"
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


def test_stderr_is_summarised_to_the_useful_line():
    """The cause must reach the log rather than being discarded."""
    assert TNPWrapper._summarize(CUDA_FAULT) == (
        "RuntimeError: CUDA error: unspecified launch failure"
    )
    assert TNPWrapper._summarize("") == ""
    assert TNPWrapper._summarize(None) == ""


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS {_name}")
    print("all tests passed")
