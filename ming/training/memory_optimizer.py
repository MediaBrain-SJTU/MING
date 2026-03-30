"""
内存优化器
提供训练过程中的显存优化功能
目标：训练显存峰值 <= 22GB
"""
import gc
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class MemoryStats:
    """内存统计"""
    allocated_gb: float
    reserved_gb: float
    max_allocated_gb: float
    max_reserved_gb: float
    free_gb: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "allocated_gb": self.allocated_gb,
            "reserved_gb": self.reserved_gb,
            "max_allocated_gb": self.max_allocated_gb,
            "max_reserved_gb": self.max_reserved_gb,
            "free_gb": self.free_gb
        }


class MemoryOptimizer:
    """
    内存优化器
    
    功能：
    - 显存监控
    - 梯度检查点优化
    - 混合精度训练
    - 内存碎片整理
    - 批次大小自适应调整
    """
    
    def __init__(
        self,
        max_memory_gb: float = 22.0,
        target_memory_gb: float = 20.0,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision: bool = True,
        enable_cpu_offload: bool = False
    ):
        """
        初始化内存优化器
        
        Args:
            max_memory_gb: 最大显存限制（GB）
            target_memory_gb: 目标显存使用（GB）
            enable_gradient_checkpointing: 是否启用梯度检查点
            enable_mixed_precision: 是否启用混合精度
            enable_cpu_offload: 是否启用CPU卸载
        """
        self.max_memory_gb = max_memory_gb
        self.target_memory_gb = target_memory_gb
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_cpu_offload = enable_cpu_offload
        
        self._memory_history: List[MemoryStats] = []
        self._optimal_batch_size: Optional[int] = None
    
    def get_memory_stats(self) -> MemoryStats:
        """获取当前内存统计"""
        if not torch.cuda.is_available():
            return MemoryStats(0, 0, 0, 0, 0)
        
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free = total_memory - allocated
        
        return MemoryStats(
            allocated_gb=allocated,
            reserved_gb=reserved,
            max_allocated_gb=max_allocated,
            max_reserved_gb=max_reserved,
            free_gb=free
        )
    
    def check_memory_constraint(self) -> Tuple[bool, float]:
        """
        检查是否满足内存约束
        
        Returns:
            (是否满足约束, 当前显存使用GB)
        """
        stats = self.get_memory_stats()
        self._memory_history.append(stats)
        
        return stats.max_allocated_gb <= self.max_memory_gb, stats.max_allocated_gb
    
    def optimize_model(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None
    ) -> nn.Module:
        """
        优化模型以减少显存使用
        
        Args:
            model: 原始模型
            tokenizer: 分词器
            
        Returns:
            优化后的模型
        """
        if self.enable_gradient_checkpointing:
            model = self._apply_gradient_checkpointing(model)
        
        if self.enable_mixed_precision:
            model = self._apply_mixed_precision(model)
        
        return model
    
    def _apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """应用梯度检查点"""
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        elif hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing_enable"):
            model.model.gradient_checkpointing_enable()
        else:
            from torch.utils.checkpoint import checkpoint
            for module in model.modules():
                if isinstance(module, nn.TransformerEncoderLayer):
                    module.forward = lambda *args, m=module, **kwargs: checkpoint(
                        m._original_forward, *args, **kwargs
                    )
                    module._original_forward = module.forward
        
        return model
    
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """应用混合精度"""
        model = model.to(torch.float16)
        return model
    
    def find_optimal_batch_size(
        self,
        model: nn.Module,
        max_batch_size: int = 32,
        sequence_length: int = 512,
        growth_factor: float = 2.0
    ) -> int:
        """
        寻找最优批次大小
        
        Args:
            model: 模型
            max_batch_size: 最大批次大小
            sequence_length: 序列长度
            growth_factor: 增长因子
            
        Returns:
            最优批次大小
        """
        if self._optimal_batch_size is not None:
            return self._optimal_batch_size
        
        model.eval()
        device = next(model.parameters()).device
        
        low, high = 1, max_batch_size
        optimal = 1
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                self._clear_memory()
                
                dummy_input = torch.randint(
                    0, 1000, (mid, sequence_length), device=device
                )
                
                with torch.no_grad():
                    _ = model(dummy_input)
                
                stats = self.get_memory_stats()
                
                if stats.allocated_gb < self.target_memory_gb:
                    optimal = mid
                    low = mid + 1
                else:
                    high = mid - 1
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    high = mid - 1
                    self._clear_memory()
                else:
                    raise e
        
        self._optimal_batch_size = optimal
        return optimal
    
    def _clear_memory(self) -> None:
        """清理显存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def optimize_for_training(
        self,
        model: nn.Module,
        batch_size: int,
        gradient_accumulation_steps: int = 1
    ) -> Dict[str, Any]:
        """
        为训练优化配置
        
        Args:
            model: 模型
            batch_size: 批次大小
            gradient_accumulation_steps: 梯度累积步数
            
        Returns:
            优化配置字典
        """
        self._clear_memory()
        
        model = self.optimize_model(model)
        
        optimal_batch_size = self.find_optimal_batch_size(model)
        
        if batch_size > optimal_batch_size:
            warnings.warn(
                f"Batch size {batch_size} exceeds optimal {optimal_batch_size}. "
                f"Adjusting gradient accumulation steps."
            )
            new_gradient_accumulation = (batch_size // optimal_batch_size) * gradient_accumulation_steps
            effective_batch_size = optimal_batch_size
        else:
            new_gradient_accumulation = gradient_accumulation_steps
            effective_batch_size = batch_size
        
        stats = self.get_memory_stats()
        
        return {
            "model": model,
            "batch_size": effective_batch_size,
            "gradient_accumulation_steps": new_gradient_accumulation,
            "memory_stats": stats.to_dict(),
            "within_constraint": stats.max_allocated_gb <= self.max_memory_gb
        }
    
    def get_memory_recommendations(self) -> List[str]:
        """
        获取内存优化建议
        
        Returns:
            建议列表
        """
        recommendations = []
        stats = self.get_memory_stats()
        
        if stats.max_allocated_gb > self.max_memory_gb * 0.9:
            recommendations.append(
                f"显存使用接近上限 ({stats.max_allocated_gb:.2f}GB / {self.max_memory_gb}GB)，"
                "建议减小批次大小或启用更多优化"
            )
        
        if not self.enable_gradient_checkpointing:
            recommendations.append(
                "建议启用梯度检查点以减少激活值显存占用"
            )
        
        if not self.enable_mixed_precision:
            recommendations.append(
                "建议启用混合精度训练（FP16/BF16）以减少显存占用"
            )
        
        if stats.reserved_gb > stats.allocated_gb * 1.5:
            recommendations.append(
                "存在显存碎片，建议调用 empty_cache() 或重启训练"
            )
        
        return recommendations
    
    def monitor_memory(
        self,
        interval_seconds: float = 1.0,
        callback: Optional[callable] = None
    ) -> None:
        """
        监控显存使用
        
        Args:
            interval_seconds: 监控间隔
            callback: 回调函数
        """
        import time
        
        while True:
            stats = self.get_memory_stats()
            self._memory_history.append(stats)
            
            if callback:
                callback(stats)
            
            if stats.allocated_gb > self.max_memory_gb:
                warnings.warn(
                    f"显存使用超过限制: {stats.allocated_gb:.2f}GB > {self.max_memory_gb}GB"
                )
            
            time.sleep(interval_seconds)
    
    def get_memory_history(self) -> List[MemoryStats]:
        """获取内存历史记录"""
        return self._memory_history.copy()
    
    def estimate_memory_requirement(
        self,
        model_params: int,
        batch_size: int,
        sequence_length: int,
        precision: str = "fp16"
    ) -> Dict[str, float]:
        """
        估算显存需求
        
        Args:
            model_params: 模型参数量
            batch_size: 批次大小
            sequence_length: 序列长度
            precision: 精度 (fp32, fp16, bf16)
            
        Returns:
            显存估算字典
        """
        bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2}.get(precision, 2)
        
        model_memory = model_params * bytes_per_param / (1024 ** 3)
        
        optimizer_memory = model_memory * 2
        
        hidden_size = 4096
        num_layers = 32
        activation_memory = (
            batch_size * sequence_length * hidden_size * num_layers * bytes_per_param
        ) / (1024 ** 3)
        
        if self.enable_gradient_checkpointing:
            activation_memory *= 0.3
        
        total_memory = model_memory + optimizer_memory + activation_memory
        
        return {
            "model_memory_gb": model_memory,
            "optimizer_memory_gb": optimizer_memory,
            "activation_memory_gb": activation_memory,
            "total_estimated_gb": total_memory,
            "within_constraint": total_memory <= self.max_memory_gb
        }
    
    def apply_dynamic_quantization(
        self,
        model: nn.Module,
        quantization_type: str = "int8"
    ) -> nn.Module:
        """
        应用动态量化
        
        Args:
            model: 模型
            quantization_type: 量化类型
            
        Returns:
            量化后的模型
        """
        if quantization_type == "int8":
            model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
        
        return model
    
    def offload_to_cpu(
        self,
        model: nn.Module,
        offload_layers: Optional[List[str]] = None
    ) -> nn.Module:
        """
        将部分层卸载到CPU
        
        Args:
            model: 模型
            offload_layers: 要卸载的层名称列表
            
        Returns:
            处理后的模型
        """
        if not self.enable_cpu_offload:
            return model
        
        if offload_layers is None:
            return model
        
        for name, module in model.named_modules():
            if any(layer_name in name for layer_name in offload_layers):
                module.to("cpu")
        
        return model


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    获取GPU内存信息
    
    Returns:
        GPU内存信息字典
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    return {
        "available": True,
        "device_name": props.name,
        "total_memory_gb": props.total_memory / (1024 ** 3),
        "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "multi_processor_count": props.multi_processor_count
    }


def clear_gpu_memory() -> None:
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
