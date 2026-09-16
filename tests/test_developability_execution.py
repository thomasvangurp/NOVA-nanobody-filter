"""Exercise the real service class without importing GPU/application modules."""

import ast
import asyncio
import hashlib
import types
import typing
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "metanano/services/developability_service.py"
tree = ast.parse(SOURCE.read_text())
class_node = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "DevelopabilityService")


def service_class(filter_class):
    namespace = {
        "asyncio": asyncio, "hashlib": hashlib,
        "DevelopabilityConfig": object, "DevelopabilityResult": object,
        "DevelopabilityFilter": filter_class, "AsyncServiceManager": object,
        **{name: getattr(typing, name) for name in ("Any", "Dict", "List", "Optional", "Tuple")},
    }
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["DevelopabilityService"]


class Manager:
    task_timeout = 300
    batch_size = 50

    def __init__(self):
        self.tnp_semaphore = asyncio.Semaphore(1)

    async def initialize(self):
        pass


def test_logging_preserves_existing_wrapper_call_api():
    received = []

    class Filter:
        def __init__(self, _):
            pass

        def compute_tnp_profile(self, sequence):
            received.append(sequence)
            return {"flags": {"L": "green"}}

    async def run():
        service = service_class(Filter)(None, Manager())
        result = await service.compute_tnp_profile_async("QVQL")
        assert result["flags"]["L"] == "green"

    asyncio.run(run())
    assert received == ["QVQL"]


def test_batch_timeout_is_typed_compute_error_and_does_not_publish_exception():
    class Filter:
        def __init__(self, _):
            pass

    async def run():
        service = service_class(Filter)(None, Manager())

        async def fail(self, sequence):
            raise TimeoutError("UNTRUSTED_MARKER")

        service.analyze_async = types.MethodType(fail, service)
        result = (await service.analyze_batch_async(["QVQL"]))[0]
        assert result["error"] is True
        assert result["error_kind"] == "timeout"
        assert result["sequence_sha256"] == hashlib.sha256(b"QVQL").hexdigest()
        assert "UNTRUSTED_MARKER" not in str(result)

    asyncio.run(run())


def test_no_profile_is_compute_error_not_biological_verdict():
    class Filter:
        def __init__(self, _):
            pass

        def compute_tnp_profile(self, sequence):
            return None

    async def run():
        service = service_class(Filter)(None, Manager())
        result = await service.analyze_async("QVQL")
        assert result["passed"] is False
        assert result["error"] is True
        assert result["error_kind"] == "no_profile"

    asyncio.run(run())


def test_biological_red_flags_still_reject_without_error_flag():
    class Filter:
        def __init__(self, _):
            pass

        def compute_tnp_profile(self, sequence):
            return {"flags": {"L": "red"}}

        def check_red_region(self, profile):
            return False, ["L red"]

    async def run():
        service = service_class(Filter)(None, Manager())
        result = await service.analyze_async("QVQL")
        assert result["passed"] is False
        assert "error" not in result
        assert result["red_flags"] == ["L red"]

    asyncio.run(run())
