from typing import Iterable, List, Optional, Tuple
from transformers import Qwen2Config

from vllm.config import CacheConfig, LoRAConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.model.qwen2 import Qwen2DecoderLayer, Qwen2MLP, Qwen2Model, Qwen2ForCausalLM
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

class MoLoRAQwenMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

class MoLoRAQwenDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        self.mlp = MoLoRAQwenMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )

class MoLoRAQwenModel(Qwen2Model):
    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__(config, cache_config, quant_config)
        self.layers = nn.ModuleList([
            MoLoRAQwenDecoderLayer(config, layer_idx, cache_config, quant_config)
            for layer_idx in range(config.num_hidden_layers)
        ])

class MoLoRAQwenForCausalLM(Qwen2ForCausalLM):
    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__(config, cache_config, quant_config, lora_config)
        self.model = MoLoRAQwenModel(config, cache_config, quant_config)
    