"""
References / 参考:
    - docs/en/README.md: Section 1.3 - Developability Filter
    - metanano/filters/developability.py: DevelopabilityFilter
    - metanano/config.py: DevelopabilityConfig

File / 文件:
    - metanano/services/developability_service.py

Overview / 概述:
    Async developability service with semaphore-based concurrency control.
    基于信号量的异步可开发性服务并发控制。

    Uses TNP CLI subprocess for profiling.
    使用 TNP CLI 子进程进行分析。

Consumers / 调用方:
    - metanano/services/__init__.py
    - metanano/routes/developability_routes.py
    - metanano/pipeline.py
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from metanano.config import DevelopabilityConfig
from metanano.filters.developability import DevelopabilityFilter, DevelopabilityResult
from metanano.services.async_manager import AsyncServiceManager, get_service_manager


class DevelopabilityService:
    """
    Async service for developability filter operations.
    可开发性过滤器操作的异步服务。

    Wraps DevelopabilityFilter with async execution and semaphore control.
    Uses semaphore to limit concurrent TNP subprocess calls.
    用异步执行和信号量控制封装 DevelopabilityFilter。
    使用信号量限制并发 TNP 子进程调用。

    Example / 示例:
        >>> service = DevelopabilityService(config)
        >>> result = await service.analyze_async(sequence)

    Consumers / 调用方:
        - metanano/routes/developability_routes.py
        - metanano/pipeline.py
    """

    def __init__(
        self,
        config: DevelopabilityConfig,
        manager: Optional[AsyncServiceManager] = None,
    ) -> None:
        """
        Initialize the developability service.
        初始化可开发性服务。

        Args / 参数:
            config (DevelopabilityConfig): Developability filter configuration.
                可开发性过滤器配置。
            manager (Optional[AsyncServiceManager]): Service manager for semaphores.
                用于信号量的服务管理器。
        """
        self.config = config
        self._filter = DevelopabilityFilter(config)
        self._manager = manager

    @property
    def manager(self) -> AsyncServiceManager:
        """Get or create service manager. / 获取或创建服务管理器。"""
        if self._manager is None:
            self._manager = get_service_manager()
        return self._manager

    async def compute_tnp_profile_async(
        self,
        sequence: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Async compute TNP profile for sequence.
        异步计算序列的 TNP 分析结果。

        Uses semaphore to limit concurrent TNP subprocess calls.
        使用信号量限制并发 TNP 子进程调用。

        Args / 参数:
            sequence (str): The nanobody sequence. / 纳米抗体序列。

        Returns / 返回:
            Optional[Dict[str, Any]]: TNP profile or None if failed.
        """
        await self.manager.initialize()
        async with self.manager.tnp_semaphore:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._filter.compute_tnp_profile,
                    sequence,
                ),
                timeout=self.manager.task_timeout,
            )

    async def check_red_region_async(
        self,
        profile: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Async check Red Region criteria.
        异步检查红区标准。

        Args / 参数:
            profile (Dict[str, Any]): TNP profile results. / TNP 分析结果。

        Returns / 返回:
            Tuple[bool, List[str]]: (passed, list_of_red_flags)
        """
        # This is a pure computation, no need for semaphore
        # 这是纯计算，不需要信号量
        return self._filter.check_red_region(profile)

    async def analyze_async(self, sequence: str) -> Dict[str, Any]:
        """
        Perform complete async developability analysis.
        执行完整的异步可开发性分析。

        Args / 参数:
            sequence (str): The sequence to analyze. / 要分析的序列。

        Returns / 返回:
            Dict[str, Any]: Complete analysis result. / 完整的分析结果。
        """
        await self.manager.initialize()

        # Compute TNP profile
        # 计算 TNP 分析结果
        profile = await self.compute_tnp_profile_async(sequence)
        if not profile:
            # No profile means the TNP tool could not run to completion, which is
            # a statement about the host and not about the molecule. "error" lets
            # callers tell an infrastructure failure apart from a sequence that
            # was measured and found undevelopable.
            # 没有 profile 意味着 TNP 工具无法运行完成，这是关于主机的问题，
            # 而不是关于分子的问题。"error" 让调用方区分基础设施故障与
            # 已测量且不可开发的序列。
            return {
                "passed": False,
                "error": True,
                "reason": "Failed to compute TNP profile. / 无法计算 TNP 分析结果。",
            }

        # Check Red Region criteria
        # 检查红区标准
        passed, red_flags = await self.check_red_region_async(profile)

        result: Dict[str, Any] = {
            "passed": passed,
            "total_cdr_length": profile.get("total_cdr_length"),
            "cdr3_length": profile.get("cdr3_length"),
            "cdr3_compactness": profile.get("cdr3_compactness"),
            "surface_hydrophobic_patches": profile.get("surface_hydrophobic_patches"),
            "positive_charge_patches": profile.get("positive_charge_patches"),
            "negative_charge_patches": profile.get("negative_charge_patches"),
            "flags": profile.get("flags"),
        }

        if red_flags:
            result["red_flags"] = red_flags
            result["reason"] = "; ".join(red_flags)

        return result

    async def analyze_batch_async(
        self,
        sequences: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple sequences concurrently.
        并发分析多个序列。

        Args / 参数:
            sequences (List[str]): Sequences to analyze. / 要分析的序列。

        Returns / 返回:
            List[Dict[str, Any]]: Results for each sequence. / 每个序列的结果。
        """
        await self.manager.initialize()

        # Process in batches respecting semaphore limits
        # 按批次处理，遵守信号量限制
        batch_size = self.manager.batch_size
        results = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[self.analyze_async(seq) for seq in batch],
                return_exceptions=True,
            )

            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({
                        "passed": False,
                        "reason": f"Error: {str(result)}",
                    })
                else:
                    results.append(result)

        return results

