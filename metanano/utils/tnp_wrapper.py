"""
References / 参考:
    - docs/en/README.md: Section 1.3 - Developability Filter
    - docs/en/TODO.md: Section 0.1 - TNP Integration
    - TNP GitHub: https://github.com/oxpig/TNP

File / 文件:
    - metanano/utils/tnp_wrapper.py

Overview / 概述:
    Wrapper for TNP (Therapeutic Nanobody Profiler) CLI tool.
    TNP（治疗性纳米抗体分析器）CLI 工具的封装器。

    TNP CLI usage pattern:
    TNP CLI 使用模式：
        TNP --name <name> --output <dir> --seq <sequence>

    Output format (JSON): TNP_Results_SingleSeqEntry_<name>.json
    输出格式（JSON）：TNP_Results_SingleSeqEntry_<name>.json

    Sample output structure:
    示例输出结构：
    {
        "my_sequence": {
            "name": "my_sequence",
            "Total CDR Length": 29,
            "CDR3 Length": 13,
            "CDR3 Compactness": 0.9288582368386492,
            "PSH": 88.7932,
            "PPC": 0.0505,
            "PNC": 0.3852,
            "Flags": {"L": "green", "L3": "green", ...}
        }
    }

Consumers / 调用方:
    - metanano/utils/__init__.py
    - metanano/filters/developability.py
"""

import json
import hashlib
import logging
import re
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# Only scientific source locations are loggable. Full paths and source lines
# can contain sensitive values, so do not forward them from a raw traceback.
_TRACEBACK_COMPONENTS = {
    "TNP": "TNP",
    "CDR_Assigner.py": "tnp_cdr_assigner",
    "CDR3_Conf_Assigner.py": "tnp_cdr3_compactness",
    "Hydrophobicity_and_Charge_Assigner.py": "tnp_surface_properties",
    "PDBUtils.py": "tnp_pdb_utils",
    "ABodyBuilder2.py": "immunebuilder_abodybuilder2",
    "NanoBodyBuilder2.py": "immunebuilder_nanobodybuilder2",
    "NBBuilder2.py": "immunebuilder_nbbuilder2",
    "sequence_checks.py": "immunebuilder_sequence_checks",
    "anarci.py": "anarci",
}


class TNPResult(BaseModel):
    """
    Parsed result from TNP CLI output.
    从 TNP CLI 输出解析的结果。

    Attributes / 属性:
        name: Sequence name / 序列名称
        total_cdr_length: Total length of all CDRs (L) / 所有 CDR 的总长度
        cdr3_length: Length of CDR3 (L3) / CDR3 的长度
        cdr3_compactness: CDR3 compactness score (C) / CDR3 紧凑度分数
        psh: Surface Hydrophobic Patches score / 表面疏水性斑块分数
        ppc: Positive Charge Patches score / 正电荷斑块分数
        pnc: Negative Charge Patches score / 负电荷斑块分数
        flags: Flag colors for each property / 每个属性的标志颜色

    Consumers / 调用方:
        - metanano/filters/developability.py
    """

    name: str = Field(
        ...,
        description="Sequence name / 序列名称",
    )
    total_cdr_length: int = Field(
        ...,
        description="Total length of all CDRs (L) / 所有 CDR 的总长度",
    )
    cdr3_length: int = Field(
        ...,
        description="Length of CDR3 (L3) / CDR3 的长度",
    )
    cdr3_compactness: float = Field(
        ...,
        description="CDR3 compactness score (C) / CDR3 紧凑度分数",
    )
    psh: float = Field(
        ...,
        description="Surface Hydrophobic Patches score (PSH) / 表面疏水性斑块分数",
    )
    ppc: float = Field(
        ...,
        description="Positive Charge Patches score (PPC) / 正电荷斑块分数",
    )
    pnc: float = Field(
        ...,
        description="Negative Charge Patches score (PNC) / 负电荷斑块分数",
    )
    flags: Dict[str, str] = Field(
        default_factory=dict,
        description="Flag colors for each property (green/yellow/red) / "
        "每个属性的标志颜色（绿色/黄色/红色）",
    )

    def to_profile_dict(self) -> Dict[str, Any]:
        """
        Convert to profile dict format expected by DevelopabilityFilter.
        转换为 DevelopabilityFilter 期望的 profile 字典格式。

        Returns / 返回:
            Dict[str, Any]: Profile dictionary with standard keys.
                具有标准键的 profile 字典。
        """
        return {
            "total_cdr_length": self.total_cdr_length,
            "cdr3_length": self.cdr3_length,
            "cdr3_compactness": self.cdr3_compactness,
            "surface_hydrophobic_patches": self.psh,
            "positive_charge_patches": self.ppc,
            "negative_charge_patches": self.pnc,
            "flags": self.flags,
        }


class TNPWrapper:
    """
    Wrapper for TNP (Therapeutic Nanobody Profiler) CLI tool.
    TNP（治疗性纳米抗体分析器）CLI 工具的封装器。

    Example / 示例:
        >>> tnp = TNPWrapper()
        >>> result = tnp.profile(sequence)
        >>> if result:
        ...     print(f"CDR3 Length: {result.cdr3_length}")

    Consumers / 调用方:
        - metanano/filters/developability.py
    """

    def __init__(
        self,
        tnp_executable: str = "TNP",
        temp_dir: Optional[Path] = None,
        max_attempts: int = 3,
    ) -> None:
        """
        Initialize the TNP wrapper.
        初始化 TNP 封装器。

        Args / 参数:
            tnp_executable (str): Path or name of TNP executable (default: "TNP").
                TNP 可执行文件的路径或名称（默认："TNP"）。
            temp_dir (Optional[Path]): Temporary directory for output files.
                If None, uses system temp directory.
                输出文件的临时目录。如果为 None，使用系统临时目录。

        References / 参考:
            - TNP GitHub: https://github.com/oxpig/TNP
        """
        self.tnp_executable = tnp_executable
        self.temp_dir = temp_dir
        self.max_attempts = max(1, int(max_attempts))

    def _check_tnp_available(self) -> bool:
        """
        Check if TNP executable is available.
        检查 TNP 可执行文件是否可用。

        Returns / 返回:
            bool: True if TNP is available, False otherwise.
                如果 TNP 可用则返回 True，否则返回 False。
        """
        return shutil.which(self.tnp_executable) is not None

    def profile(self, sequence: str, name: Optional[str] = None) -> Optional[TNPResult]:
        """
        Run TNP profiling on a sequence.
        对序列运行 TNP 分析。

        Args / 参数:
            sequence (str): The nanobody amino acid sequence.
                纳米抗体氨基酸序列。
            name (Optional[str]): Name for the sequence. If None, generates UUID.
                序列的名称。如果为 None，生成 UUID。

        Returns / 返回:
            Optional[TNPResult]: TNP result or None if profiling failed.
                TNP 结果，如果分析失败则返回 None。

        Example / 示例:
            >>> tnp = TNPWrapper()
            >>> result = tnp.profile("QVQLVQSGVEVK...")
            >>> print(result.cdr3_length)
            13

        Consumers / 调用方:
            - metanano/filters/developability.py: DevelopabilityFilter.compute_tnp_profile
        """
        started = time.monotonic()
        context = {
            "schema": 1,
            "profile_id": uuid.uuid4().hex,
            "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
            "sequence_length": len(sequence),
            "max_attempts": self.max_attempts,
        }
        self._diagnostic(context, event="profile_started", attempt=0)
        if not self._check_tnp_available():
            self._diagnostic(context, event="profile_exhausted", attempt=0,
                             categories=["executable_missing"], elapsed_s=0.0)
            return None

        # TNP delegates structure prediction to NanoBodyBuilder2 (ImmuneBuilder)
        # on the GPU, and that can fault transiently -- observed as
        # "RuntimeError: CUDA error: unspecified launch failure". TNP catches it
        # and exits without a profile. That is a property of the host at that
        # moment, not of the sequence: the same sequence profiles cleanly on a
        # retry. Retrying here keeps a transient fault from being reported as a
        # verdict about the molecule.
        # TNP 将结构预测委托给 GPU 上的 NanoBodyBuilder2（ImmuneBuilder），
        # 它可能出现瞬时故障。这是主机当时的状态问题，而非序列的问题：
        # 同一序列在重试时可以正常分析。
        for attempt in range(1, self.max_attempts + 1):
            result = self._run_once(sequence, name, attempt, context=context)
            if result is not None:
                self._diagnostic(context, event="profile_succeeded", attempt=attempt,
                                 elapsed_s=round(time.monotonic() - started, 3))
                return result

        self._diagnostic(context, event="profile_exhausted", attempt=self.max_attempts,
                         elapsed_s=round(time.monotonic() - started, 3))
        return None

    @staticmethod
    def _diagnostic(context: Dict[str, Any], **fields: Any) -> None:
        """Only fixed categories, scientific digests and numeric data reach logs.

        Never forward CLI output, exception strings, paths or environment values.
        Each profile has its own context, even on a shared concurrent wrapper.
        """
        event = {**context, **fields}
        logger.warning("TNP_DIAGNOSTIC %s", json.dumps(event, sort_keys=True),
                       extra={"tnp_diagnostic": event})

    @staticmethod
    def _failure_categories(*streams: Any) -> list[str]:
        text = "\n".join(
            stream.decode("utf-8", errors="replace") if isinstance(stream, bytes)
            else stream if isinstance(stream, str) else ""
            for stream in streams
        ).lower()
        patterns = {
            "cuda_launch_failure": ("unspecified launch failure",),
            "cuda_oom": ("cuda out of memory", "cuda error: out of memory"),
            "cuda_illegal_memory": ("illegal memory access",),
            "cuda_device_assert": ("device-side assert",),
            "model_prediction_failed": ("failed to generate a model",),
            "openmm_error": ("openmmexception", "openmm error"),
            "import_error": ("modulenotfounderror", "importerror"),
            "none_subscript": ("typeerror: 'nonetype' object is not subscriptable",),
        }
        return [category for category, needles in patterns.items()
                if any(needle in text for needle in needles)]

    @staticmethod
    def _traceback_locations(*streams: Any) -> list[dict[str, Any]]:
        locations = []
        for stream in streams:
            if isinstance(stream, bytes):
                stream = stream.decode("utf-8", errors="replace")
            if not isinstance(stream, str):
                continue
            for match in re.finditer(r'File "([^"\r\n]+)", line ([0-9]{1,7})', stream):
                filename = match.group(1).replace("\\", "/").rsplit("/", 1)[-1]
                component = _TRACEBACK_COMPONENTS.get(filename)
                if component is not None:
                    locations.append({"component": component, "line": int(match.group(2))})
        return locations[-16:]

    def _run_once(
        self,
        sequence: str,
        name: Optional[str],
        attempt: int,
        *, timeout: float = 300.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[TNPResult]:
        """
        Run TNP once, recording allowlisted per-call diagnostics.

        Returns / 返回:
            Optional[TNPResult]: Parsed result, or None if this attempt failed.
                解析后的结果，如果本次尝试失败则返回 None。
        """
        context = context or {
            "schema": 1, "profile_id": uuid.uuid4().hex,
            "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
            "sequence_length": len(sequence),
            "max_attempts": self.max_attempts,
        }
        started = time.monotonic()
        self._diagnostic(context, event="attempt_started", attempt=attempt,
                         timeout_s=round(timeout, 3))
        # Generate unique name if not provided
        # 如果未提供，生成唯一名称
        seq_name = name or f"seq_{uuid.uuid4().hex[:8]}"
        if attempt > 1:
            # A fresh name per attempt keeps TNP's output files from colliding.
            # 每次尝试使用新名称，避免 TNP 输出文件冲突。
            seq_name = f"{seq_name}_r{attempt}"

        # Create temporary directory for output
        # 创建输出的临时目录
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as tmpdir:
            output_dir = Path(tmpdir)

            # Run TNP CLI
            # 运行 TNP CLI
            # Command: TNP --name <name> --output <dir> --seq <sequence>
            cmd = [
                self.tnp_executable,
                "--name", seq_name,
                "--output", str(output_dir),
                "--seq", sequence,
            ]

            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=timeout,
                )
            except subprocess.CalledProcessError as e:
                categories = self._failure_categories(e.stdout, e.stderr)
                self._diagnostic(context, event="attempt_failed", attempt=attempt,
                                 returncode=e.returncode,
                                 categories=categories or ["process_failed"],
                                 traceback_locations=self._traceback_locations(e.stdout, e.stderr),
                                 elapsed_s=round(time.monotonic() - started, 3))
                return None
            except subprocess.TimeoutExpired as e:
                self._diagnostic(context, event="attempt_failed", attempt=attempt,
                                 categories=["timeout"] + self._failure_categories(e.stdout, e.stderr),
                                 traceback_locations=self._traceback_locations(e.stdout, e.stderr),
                                 timeout_s=round(timeout, 3),
                                 elapsed_s=round(time.monotonic() - started, 3))
                return None
            except FileNotFoundError:
                self._diagnostic(context, event="attempt_failed", attempt=attempt,
                                 categories=["executable_missing"],
                                 elapsed_s=round(time.monotonic() - started, 3))
                return None
            except OSError:
                self._diagnostic(context, event="attempt_failed", attempt=attempt,
                                 categories=["process_start_failed"],
                                 elapsed_s=round(time.monotonic() - started, 3))
                return None

            # Parse output JSON
            # 解析输出 JSON
            parsed = self._parse_output(output_dir, seq_name)
            if parsed is None:
                exists = (output_dir / f"TNP_Results_SingleSeqEntry_{seq_name}.json").exists()
                categories = self._failure_categories(completed.stdout, completed.stderr)
                self._diagnostic(context, event="attempt_failed", attempt=attempt,
                                 returncode=completed.returncode,
                                 categories=categories + ["profile_invalid" if exists else "profile_missing"],
                                 traceback_locations=self._traceback_locations(completed.stdout, completed.stderr),
                                 elapsed_s=round(time.monotonic() - started, 3))
            else:
                self._diagnostic(context, event="attempt_succeeded", attempt=attempt,
                                 returncode=completed.returncode,
                                 elapsed_s=round(time.monotonic() - started, 3))
            return parsed

    def _parse_output(
        self,
        output_dir: Path,
        seq_name: str,
    ) -> Optional[TNPResult]:
        """
        Parse TNP output JSON file.
        解析 TNP 输出 JSON 文件。

        Args / 参数:
            output_dir (Path): Directory containing TNP output.
                包含 TNP 输出的目录。
            seq_name (str): Name of the sequence (used in filename).
                序列的名称（用于文件名）。

        Returns / 返回:
            Optional[TNPResult]: Parsed result or None if parsing failed.
                解析后的结果，如果解析失败则返回 None。
        """
        # Expected output file: TNP_Results_SingleSeqEntry_<name>.json
        # 预期输出文件：TNP_Results_SingleSeqEntry_<name>.json
        output_file = output_dir / f"TNP_Results_SingleSeqEntry_{seq_name}.json"

        if not output_file.exists():
            return None

        try:
            with open(output_file, "r") as f:
                data = json.load(f)

            # Extract sequence data
            # 提取序列数据
            if seq_name not in data:
                return None

            seq_data = data[seq_name]

            # Parse fields
            # 解析字段
            return TNPResult(
                name=seq_data.get("name", seq_name),
                total_cdr_length=seq_data.get("Total CDR Length", 0),
                cdr3_length=seq_data.get("CDR3 Length", 0),
                cdr3_compactness=seq_data.get("CDR3 Compactness", 0.0),
                psh=seq_data.get("PSH", 0.0),
                ppc=seq_data.get("PPC", 0.0),
                pnc=seq_data.get("PNC", 0.0),
                flags=seq_data.get("Flags", {}),
            )

        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValidationError, OSError):
            return None

    def profile_batch(
        self,
        sequences: Dict[str, str],
    ) -> Dict[str, Optional[TNPResult]]:
        """
        Profile multiple sequences.
        分析多个序列。

        Args / 参数:
            sequences (Dict[str, str]): Mapping of name to sequence.
                名称到序列的映射。

        Returns / 返回:
            Dict[str, Optional[TNPResult]]: Results for each sequence.
                每个序列的结果。

        Consumers / 调用方:
            - metanano/validators/developability_validator.py (batch processing)
        """
        results = {}
        for name, sequence in sequences.items():
            results[name] = self.profile(sequence, name=name)
        return results
