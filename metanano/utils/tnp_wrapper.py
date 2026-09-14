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
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
        self._last_error = "unknown"

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
        if not self._check_tnp_available():
            logger.error("TNP executable %r not found on PATH", self.tnp_executable)
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
            result = self._run_once(sequence, name, attempt)
            if result is not None:
                if attempt > 1:
                    logger.info("TNP succeeded on attempt %d", attempt)
                return result

        logger.warning(
            "TNP produced no profile after %d attempt(s); last failure: %s",
            self.max_attempts,
            self._last_error,
        )
        return None

    def _run_once(
        self,
        sequence: str,
        name: Optional[str],
        attempt: int,
    ) -> Optional[TNPResult]:
        """
        Run TNP once, recording the failure cause on self._last_error.
        运行一次 TNP，将失败原因记录在 self._last_error 上。

        Returns / 返回:
            Optional[TNPResult]: Parsed result, or None if this attempt failed.
                解析后的结果，如果本次尝试失败则返回 None。
        """
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
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300,  # 5 minutes timeout
                )
            except subprocess.CalledProcessError as e:
                # TNP exits non-zero when it could not model the sequence, so its
                # stderr is the only record of why. Keep it out of the return
                # value but put it in the log.
                # TNP 无法建模时以非零状态退出，其 stderr 是唯一的原因记录。
                self._last_error = self._summarize(e.stderr) or f"exit {e.returncode}"
                logger.warning(
                    "TNP attempt %d failed (exit %s): %s",
                    attempt, e.returncode, self._last_error,
                )
                return None
            except subprocess.TimeoutExpired:
                self._last_error = "timed out after 300s"
                logger.warning("TNP attempt %d timed out", attempt)
                return None
            except FileNotFoundError:
                self._last_error = f"executable {self.tnp_executable!r} not found"
                logger.error("TNP executable disappeared mid-run")
                return None

            # Parse output JSON
            # 解析输出 JSON
            parsed = self._parse_output(output_dir, seq_name)
            if parsed is None:
                self._last_error = "TNP exited cleanly but wrote no usable profile"
                logger.warning("TNP attempt %d: %s", attempt, self._last_error)
            return parsed

    @staticmethod
    def _summarize(stderr: Optional[str], limit: int = 300) -> str:
        """
        Reduce a TNP traceback to its final, informative line.
        将 TNP 的回溯信息缩减为最后一行有效信息。
        """
        if not stderr:
            return ""
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        for line in reversed(lines):
            if not line.startswith('File "') and "^" not in line:
                return line[:limit]
        return lines[-1][:limit] if lines else ""

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

        except (json.JSONDecodeError, KeyError, TypeError):
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




