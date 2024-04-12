from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from .utils import MoLoRALinear

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2MLP, Qwen2Model, Qwen2ForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPast
import torch.nn as nn
import torch 
import torch.nn.functional as F 
from typing import Optional, Tuple, Union, List
import warnings
from transformers.utils import logging
from dataclasses import dataclass


logger = logging.get_logger(__name__)

class MoLoRAQwen2Config(Qwen2Config):
    def __init__(self, vocab_size=151936, hidden_size=4096, intermediate_size=22016, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32, hidden_act="silu", max_position_embeddings=32768, initializer_range=0.02, rms_norm_eps=0.000001, use_cache=True, tie_word_embeddings=False, rope_theta=10000, use_sliding_window=False, sliding_window=4096, max_window_layers=28, attention_dropout=0, **kwargs):
        super().__init__(vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, hidden_act, max_position_embeddings, initializer_range, rms_norm_eps, use_cache, tie_word_embeddings, rope_theta, use_sliding_window, sliding_window, max_window_layers, attention_dropout, **kwargs)

@dataclass
class BaseModelOutputWithPastLogitBias(ModelOutput):
    """
    Base class for model's outputs, with past key value states and logit bias.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    logit_bias: Optional[torch.FloatTensor] = None

class MoLoRAQwenMLPDeploy(Qwen2MLP):
    def __init__(self, config):
        super().__init__(config)
        params = {
            "r": config.r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "num_experts": config.num_experts,
            "num_experts_per_token": config.num_experts_per_token,
            "share_expert": getattr(config, "share_expert", False),
            "expert_sampling": True if config.expert_selection == 'sampling' else False,
            "use_rslora": getattr(config, "use_rslora", False),
            "use_logit_sum": getattr(config, "output_logit_bias", False),
        }
        self.use_logit_sum = params['use_logit_sum']
        self.gate_proj = MoLoRALinear(self.hidden_size, self.intermediate_size, bias=False,
                                      **params)
        self.up_proj = MoLoRALinear(self.hidden_size, self.intermediate_size, bias=False, **params)
        self.down_proj = MoLoRALinear(self.intermediate_size, self.hidden_size, bias=False, **params)
    
    def forward(self, x):
        if self.use_logit_sum:
            gate_output, gate_logit_sum = self.gate_proj(x)
            up_output, up_logit_sum = self.up_proj(x)
            down_output, down_logit_sum = self.down_proj(self.act_fn(gate_output) * up_output)
            # NOTE: current average the three weights logit sum, may use a nontrivial way
            # leave for future research space
            logit_sum = (gate_logit_sum + up_logit_sum + down_logit_sum) / 3
            return down_output, logit_sum
        else:
            return super().forward(x)
        
class MoLoRAQwenMLP(Qwen2MLP):
    def __init__(self, config):
        super().__init__(config)
        self.use_logit_sum = getattr(config, "output_logit_bias", False)
    
    def forward(self, x):
        if self.use_logit_sum:
            gate_output, gate_logit_sum = self.gate_proj(x)
            up_output, up_logit_sum = self.up_proj(x)
            down_output, down_logit_sum = self.down_proj(self.act_fn(gate_output) * up_output)
            # NOTE: current stack the logit sum
            logit_sum = torch.stack([gate_logit_sum, up_logit_sum, down_logit_sum], dim=0)
            return down_output, logit_sum
        else:
            return super().forward(x)

class MoLoRAQwenDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = MoLoRAQwenMLP(config)
        
        self.config = config
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        output_logit_bias: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if output_logit_bias:
            hidden_states, logit_sum = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_logit_bias:
            outputs += (logit_sum,)
        return outputs
    

class MoLoRAQwenModel(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MoLoRAQwenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
    
    # @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_logit_bias: Optional[bool] = None,
    ) -> Union[Tuple, Union[BaseModelOutputWithPastLogitBias, BaseModelOutputWithPast]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_logit_bias = output_logit_bias if output_logit_bias is not None else self.config.output_logit_bias
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_logit_bias = () if output_logit_bias else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    output_logit_bias,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    output_logit_bias=output_logit_bias
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if output_logit_bias:
                all_logit_bias += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_logit_bias] if v is not None)
        return BaseModelOutputWithPastLogitBias(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            logit_bias=all_logit_bias,
        )
        

class MoLoRAQwenForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MoLoRAQwenModel(config)
    