"""
显存监控模块

本模块提供实时GPU显存监控和控制功能，确保训练过程中显存使用在安全范围内。
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import torch

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """显存快照数据类"""
    timestamp: float
    total_memory: int
    allocated_memory: int
    cached_memory: int
    reserved_memory: int
    utilization: float

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp,
            "total_memory_gb": self.total_memory / (1024**3),
            "allocated_memory_gb": self.allocated_memory / (1024**3),
            "cached_memory_gb": self.cached_memory / (1024**3),
            "reserved_memory_gb": self.reserved_memory / (1024**3),
            "utilization": self.utilization,
        }


class GPUMemoryTracker:
    """
    GPU显存跟踪器

    实时跟踪GPU显存使用情况，提供显存统计和报警功能。

    Attributes:
        device_id: GPU设备ID
        snapshots: 显存快照历史
        max_history: 最大历史记录数
        warning_threshold: 警告阈值（使用率）
        critical_threshold: 临界阈值（使用率）
        callbacks: 阈值回调函数
    """

    def __init__(
        self,
        device_id: int = 0,
        max_history: int = 1000,
        warning_threshold: float = 0.85,
        critical_threshold: float = 0.95,
    ):
        """
        初始化显存跟踪器

        Args:
            device_id: GPU设备ID
            max_history: 最大历史记录数
            warning_threshold: 警告阈值（0-1）
            critical_threshold: 临界阈值（0-1）
        """
        self.device_id = device_id
        self.max_history = max_history
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self.snapshots: deque = deque(maxlen=max_history)
        self.callbacks: List[Callable[[MemorySnapshot, str], None]] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._peak_memory = 0

        # 检查CUDA可用性
        self._cuda_available = torch.cuda.is_available()
        if not self._cuda_available:
            logger.warning("CUDA不可用，显存跟踪功能将受限")
        else:
            self.total_memory = torch.cuda.get_device_properties(device_id).total_memory

    def snapshot(self) -> MemorySnapshot:
        """
        拍摄当前显存快照

        Returns:
            MemorySnapshot: 显存快照对象
        """
        if not self._cuda_available:
            return MemorySnapshot(
                timestamp=time.time(),
                total_memory=0,
                allocated_memory=0,
                cached_memory=0,
                reserved_memory=0,
                utilization=0.0,
            )

        device = torch.device(f"cuda:{self.device_id}")

        with torch.cuda.device(device):
            total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(self.device_id)
            reserved_memory = torch.cuda.memory_reserved(self.device_id)
            cached_memory = reserved_memory - allocated_memory
            utilization = allocated_memory / total_memory if total_memory > 0 else 0.0

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            total_memory=total_memory,
            allocated_memory=allocated_memory,
            cached_memory=cached_memory,
            reserved_memory=reserved_memory,
            utilization=utilization,
        )

        self.snapshots.append(snapshot)

        # 更新峰值
        if allocated_memory > self._peak_memory:
            self._peak_memory = allocated_memory

        # 检查阈值
        self._check_thresholds(snapshot)

        return snapshot

    def _check_thresholds(self, snapshot: MemorySnapshot) -> None:
        """检查显存阈值并触发回调"""
        if snapshot.utilization >= self.critical_threshold:
            for callback in self.callbacks:
                callback(snapshot, "CRITICAL")
        elif snapshot.utilization >= self.warning_threshold:
            for callback in self.callbacks:
                callback(snapshot, "WARNING")

    def add_callback(
        self,
        callback: Callable[[MemorySnapshot, str], None]
    ) -> None:
        """
        添加阈值回调函数

        Args:
            callback: 回调函数，参数为(MemorySnapshot, level)
        """
        self.callbacks.append(callback)

    def start_monitoring(self, interval: float = 0.1) -> None:
        """
        启动后台监控线程

        Args:
            interval: 监控间隔（秒）
        """
        if self._monitoring:
            logger.warning("显存监控已经在运行中")
            return

        self._monitoring = True

        def monitor_loop():
            while self._monitoring:
                self.snapshot()
                time.sleep(interval)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"显存监控已启动，间隔: {interval}秒")

    def stop_monitoring(self) -> None:
        """停止后台监控线程"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
            self._monitor_thread = None
        logger.info("显存监控已停止")

    def get_peak_memory(self, in_gb: bool = True) -> float:
        """
        获取峰值显存使用量

        Args:
            in_gb: 是否以GB为单位

        Returns:
            float: 峰值显存使用量
        """
        if in_gb:
            return self._peak_memory / (1024**3)
        return self._peak_memory

    def get_current_utilization(self) -> float:
        """
        获取当前显存使用率

        Returns:
            float: 当前显存使用率（0-1）
        """
        if not self.snapshots:
            return 0.0
        return self.snapshots[-1].utilization

    def get_average_utilization(self, window: int = 10) -> float:
        """
        获取最近N个快照的平均显存使用率

        Args:
            window: 窗口大小

        Returns:
            float: 平均显存使用率
        """
        if not self.snapshots:
            return 0.0

        recent_snapshots = list(self.snapshots)[-window:]
        return sum(s.utilization for s in recent_snapshots) / len(recent_snapshots)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取显存统计信息

        Returns:
            Dict[str, Any]: 统计信息字典
        """
        if not self.snapshots:
            return {"error": "No snapshots available"}

        snapshots = list(self.snapshots)
        utilizations = [s.utilization for s in snapshots]
        allocated = [s.allocated_memory for s in snapshots]

        return {
            "peak_memory_gb": self.get_peak_memory(),
            "current_utilization": self.get_current_utilization(),
            "avg_utilization_1min": self.get_average_utilization(600),
            "avg_utilization_10s": self.get_average_utilization(100),
            "max_utilization": max(utilizations),
            "min_utilization": min(utilizations),
            "total_snapshots": len(snapshots),
            "total_memory_gb": snapshots[0].total_memory / (1024**3),
        }

    def reset_peak(self) -> None:
        """重置峰值显存记录"""
        self._peak_memory = 0
        logger.info("峰值显存记录已重置")


class MemoryMonitor:
    """
    训练过程显存监控器

    集成在训练循环中，实时监控显存使用并采取保护措施：
    1. 梯度累积步数调整
    2. 批量大小动态调整
    3. 训练暂停等待显存释放
    4. 紧急停止保护

    Attributes:
        tracker: GPUMemoryTracker实例
        max_batch_size: 最大批量大小
        min_batch_size: 最小批量大小
        current_batch_size: 当前批量大小
        gradient_accumulation_steps: 梯度累积步数
        safety_margin_gb: 安全余量（GB）
    """

    def __init__(
        self,
        device_id: int = 0,
        max_batch_size: int = 16,
        min_batch_size: int = 1,
        initial_batch_size: Optional[int] = None,
        safety_margin_gb: float = 2.0,
        warning_threshold: float = 0.85,
        critical_threshold: float = 0.92,
    ):
        """
        初始化显存监控器

        Args:
            device_id: GPU设备ID
            max_batch_size: 最大批量大小
            min_batch_size: 最小批量大小
            initial_batch_size: 初始批量大小
            safety_margin_gb: 安全余量（GB）
            warning_threshold: 警告阈值
            critical_threshold: 临界阈值
        """
        self.tracker = GPUMemoryTracker(
            device_id=device_id,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
        )

        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.current_batch_size = initial_batch_size or max_batch_size
        self.gradient_accumulation_steps = 1
        self.safety_margin_gb = safety_margin_gb

        # 状态
        self._adjustment_count = 0
        self._last_adjustment_step = 0
        self._consecutive_warnings = 0

        # 注册回调
        self.tracker.add_callback(self._memory_threshold_callback)

        logger.info(
            f"显存监控器初始化完成: batch_size={self.current_batch_size}, "
            f"范围=[{min_batch_size}, {max_batch_size}]"
        )

    def _memory_threshold_callback(
        self,
        snapshot: MemorySnapshot,
        level: str
    ) -> None:
        """显存阈值回调处理"""
        if level == "CRITICAL":
            logger.warning(
                f"显存使用危急! 使用率: {snapshot.utilization:.1%}, "
                f"已分配: {snapshot.allocated_memory / (1024**3):.2f}GB"
            )
            self._consecutive_warnings += 1
        elif level == "WARNING":
            logger.info(
                f"显存使用警告: 使用率: {snapshot.utilization:.1%}, "
                f"已分配: {snapshot.allocated_memory / (1024**3):.2f}GB"
            )

    def can_allocate_memory(self, required_memory_gb: float) -> bool:
        """
        检查是否有足够显存分配

        Args:
            required_memory_gb: 需要的显存量（GB）

        Returns:
            bool: 是否可以安全分配
        """
        if not self.tracker._cuda_available:
            return True

        snapshot = self.tracker.snapshot()
        available_memory = (
            snapshot.total_memory - snapshot.allocated_memory - snapshot.cached_memory
        )
        available_gb = available_memory / (1024**3)

        return available_gb > (required_memory_gb + self.safety_margin_gb)

    def adjust_batch_size(
        self,
        current_utilization: Optional[float] = None,
        step: int = 0
    ) -> Tuple[int, bool]:
        """
        根据显存使用情况动态调整批量大小

        Args:
            current_utilization: 当前显存使用率
            step: 当前训练步数

        Returns:
            Tuple[int, bool]: 新的批量大小和是否调整
        """
        if current_utilization is None:
            snapshot = self.tracker.snapshot()
            current_utilization = snapshot.utilization

        adjusted = False
        old_batch_size = self.current_batch_size

        # 避免过于频繁的调整
        if step - self._last_adjustment_step < 10 and self._adjustment_count > 0:
            return self.current_batch_size, False

        # 显存使用率过高，减小批量
        if current_utilization >= self.tracker.critical_threshold:
            new_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.7)
            )
            if new_batch_size < self.current_batch_size:
                self.current_batch_size = new_batch_size
                self._adjustment_count += 1
                self._last_adjustment_step = step
                adjusted = True
                logger.warning(
                    f"显存过高，减小批量: {old_batch_size} -> {new_batch_size}, "
                    f"使用率: {current_utilization:.1%}"
                )

        # 显存使用率较低，尝试增大批量
        elif current_utilization < 0.6 and self._consecutive_warnings == 0:
            new_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.3)
            )
            if new_batch_size > self.current_batch_size:
                self.current_batch_size = new_batch_size
                self._adjustment_count += 1
                self._last_adjustment_step = step
                adjusted = True
                logger.info(
                    f"显存充足，增大批量: {old_batch_size} -> {new_batch_size}, "
                    f"使用率: {current_utilization:.1%}"
                )

        # 重置连续警告计数
        if current_utilization < self.tracker.warning_threshold:
            self._consecutive_warnings = 0

        return self.current_batch_size, adjusted

    def optimize_gradient_accumulation(
        self,
        target_batch_size: int,
        current_batch_size: Optional[int] = None
    ) -> int:
        """
        计算最优梯度累积步数

        Args:
            target_batch_size: 目标有效批量大小
            current_batch_size: 当前实际批量大小

        Returns:
            int: 梯度累积步数
        """
        batch_size = current_batch_size or self.current_batch_size
        gas = max(1, (target_batch_size + batch_size - 1) // batch_size)
        self.gradient_accumulation_steps = gas
        logger.info(
            f"梯度累积步数设置: {gas}, 有效批量: {batch_size * gas}"
        )
        return gas

    def get_status_report(self) -> Dict[str, Any]:
        """
        获取显存状态报告

        Returns:
            Dict[str, Any]: 状态报告字典
        """
        tracker_stats = self.tracker.get_statistics()
        return {
            **tracker_stats,
            "current_batch_size": self.current_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "adjustment_count": self._adjustment_count,
            "consecutive_warnings": self._consecutive_warnings,
            "safety_margin_gb": self.safety_margin_gb,
        }

    def start(self, interval: float = 0.1) -> None:
        """启动监控"""
        self.tracker.start_monitoring(interval)

    def stop(self) -> None:
        """停止监控"""
        self.tracker.stop_monitoring()

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
        # 打印最终报告
        report = self.get_status_report()
        logger.info("显存监控最终报告:")
        logger.info(f"  峰值显存: {report.get('peak_memory_gb', 0):.2f}GB")
        logger.info(f"  批量调整次数: {report.get('adjustment_count', 0)}")


def get_gpu_memory_info(device_id: int = 0) -> Dict[str, Any]:
    """
    获取GPU显存信息

    Args:
        device_id: GPU设备ID

    Returns:
        Dict[str, Any]: 显存信息字典
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    props = torch.cuda.get_device_properties(device_id)

    with torch.cuda.device(device_id):
        total_memory = props.total_memory
        allocated_memory = torch.cuda.memory_allocated(device_id)
        reserved_memory = torch.cuda.memory_reserved(device_id)
        cached_memory = reserved_memory - allocated_memory
        utilization = allocated_memory / total_memory if total_memory > 0 else 0

    return {
        "device_name": props.name,
        "device_id": device_id,
        "total_memory_gb": total_memory / (1024**3),
        "allocated_memory_gb": allocated_memory / (1024**3),
        "reserved_memory_gb": reserved_memory / (1024**3),
        "cached_memory_gb": cached_memory / (1024**3),
        "free_memory_gb": (total_memory - allocated_memory) / (1024**3),
        "utilization": utilization,
        "utilization_percent": f"{utilization:.1%}",
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
    }


def clear_gpu_cache(device_id: int = 0) -> None:
    """
    清理GPU缓存

    Args:
        device_id: GPU设备ID
    """
    if not torch.cuda.is_available():
        return

    with torch.cuda.device(device_id):
        cached_before = torch.cuda.memory_reserved(device_id)
        torch.cuda.empty_cache()
        cached_after = torch.cuda.memory_reserved(device_id)

        freed = (cached_before - cached_after) / (1024**3)
        logger.info(f"清理GPU缓存，释放显存: {freed:.2f}GB")


def main():
    """测试函数"""
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过显存监控测试")
        return

    # 测试GPU信息查询
    info = get_gpu_memory_info()
    print("GPU信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 测试显存跟踪器
    tracker = GPUMemoryTracker()

    # 拍摄几次快照
    for i in range(5):
        snapshot = tracker.snapshot()
        print(f"\n快照 {i+1}:")
        print(f"  时间戳: {snapshot.timestamp:.2f}")
        print(f"  显存使用率: {snapshot.utilization:.1%}")
        print(f"  已分配: {snapshot.allocated_memory / (1024**3):.2f}GB")

    # 测试显存监控器
    with MemoryMonitor(max_batch_size=8, min_batch_size=1) as monitor:
        # 模拟一些显存使用
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            tensors = []

            for i in range(3):
                # 分配一些显存
                tensors.append(torch.randn(100, 100, 100).to(device))
                torch.cuda.synchronize()

                batch_size, adjusted = monitor.adjust_batch_size()
                print(f"\n迭代 {i+1}:")
                print(f"  当前批量: {batch_size}, 是否调整: {adjusted}")
                print(f"  峰值显存: {monitor.tracker.get_peak_memory():.2f}GB")

    # 获取统计信息
    print("\n统计信息:")
    stats = tracker.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 清理缓存
    clear_gpu_cache()


if __name__ == "__main__":
    main()
