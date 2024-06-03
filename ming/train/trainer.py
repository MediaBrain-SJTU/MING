import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler
from transformers.trainer import *
from transformers import Trainer
from transformers.trainer import  TRAINER_STATE_NAME
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import *
import numpy as np
import shutil
import deepspeed
from deepspeed import zero
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_get_local_grad, safe_get_local_fp32_param

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def maybe_zero_3_nodetach(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.clone()
    else:
        param = param.clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def get_molora_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class MINGTrainer(Trainer):

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tne_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        elif getattr(self.args, 'molora', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            # Only save Adapter
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            keys_to_match = ["mlp.gate_proj.experts", "mlp.up_proj.experts", "mlp.down_proj.experts", "mlp.switch", "mlp.gate_proj.share_experts", "mlp.up_proj.share_experts", "mlp.down_proj.share_experts"]

            weight_to_save = get_molora_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'non_lora_trainables.bin'))
            super(MINGTrainer, self)._save_checkpoint(model, trial, metrics)
        elif getattr(self.args, "use_orthogonal", False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            # Only save Adapter
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            keys_to_match = ["mlp.gate_proj.orth"]
        else:
            super(MINGTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(MINGTrainer, self)._save(output_dir, state_dict)
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
        
        ########################### Regularization ##########################
        orthogonal_loss = torch.tensor(0.).to(model.device).to(loss.dtype)
        cmp_item = 0
        for name, param in self.model.named_parameters():
            if "base" in name and "lora_A" in name:
                # find all params that start with name.split("share_experts")[0]
                prefix = name.split("base")[0]
                for name_, param_ in self.model.named_parameters():
                    if prefix + "orth" in name_ and "lora_A" in name_:
                        # print(name, name_)
                        # print(param.shape, param_.shape)
                        # print(param)
                        # print(param.shape)
                        fparam = safe_get_full_fp32_param(param=param)
                        # fparam = maybe_zero_3(param)

                        # with zero.GatheredParameters([param]):
                            # print(param.shape)
                                # fparam = param.clone()
                        # except Exception as e:
                        #     print(name, fparam)
                        #     print(e)
                        #     exit(-1)
                        param = param if fparam is None else fparam
                        fparam_ = safe_get_full_fp32_param(param=param_)
                            # fparam_ = maybe_zero_3_nodetach(param_)
                            # with zero.GatheredParameters([param_]):
                            #     fparam_ = param_.data.clone()
                        param_ = param_ if fparam_ is None else fparam_
                            # # print(param_)
                            # # print(param.shape)
                            # param.to(fparam_.dtype)
                            # print(param.shape, fparam_.shape)
                            # cmp_item += 1
                        orthogonal_loss += torch.abs(torch.mm(param.float(), param_.float().T)).sum() # [r * dim] * [dim * r]
                        # break # once find, we find the layers.x.mlp.{}_proj.base.lora_A and layers.x.mlp.{}_proj.orth.lora_A 
                    
                        # with deepspeed.zero.GatheredParameters(param):
                        #     with deepspeed.zero.GatheredParameters(param_):
                            
                        #     fparam_ = safe_get_full_fp32_param(param=param_)
                        #     param_ = param_ if fparam_ is None else fparam_
                        #         # print(name, name_)
                        #     param_ = param_.to(param.dtype)
                                # print(param.shape, param_.shape)
                                # orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum() # [r * dim] * [dim * r]
                                # if param is not None and param_ is not None:
                                #     print(param.shape, param_.shape)
                        break
                        # try:
                        # param = 
                        # orthogonal_loss += torch.abs(torch.mm(param, param.transpose(0, -1))).sum() # [r * dim] * [dim * r]

                # for name_, param_ in self.model.named_parameters():
                #     if "experts" in name_ and name.split("share_experts")[0] == name_.split("experts")[0]:
                #         # for any param.lora_A and name_.
                        
                #         orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum() # [r * dim] * [dim * r]
                #         break # target modules have been matched
        # l2-normalization for loranew_A/B
        # orthogonal_loss /
        # print(orthogonal_loss)
        lamda_1 = self.args.lamda_1


        logger.info(f"orthogonal_loss: {orthogonal_loss.item()}; accuracy_loss: {loss.item()}; λ1: {lamda_1}")


        floss = loss + orthogonal_loss * lamda_1

        ######################################################################

        # if self.do_grad_scaling:
        #     self.scaler.scale(loss).backward()
        # elif self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # elif self.deepspeed:
        #     # loss gets scaled under gradient_accumulation_steps in deepspeed
        #     loss = self.deepspeed.backward(loss)
        # else:
        #     loss.backward()

        # return loss.detach()
        if self.use_apex:
            with amp.scale_loss(floss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(floss)

        return loss.detach() 