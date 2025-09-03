import torch
from collections import OrderedDict
from copy import deepcopy
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_fsdp_plugin(fsdp_cfg, mixed_precision):
    import functools
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        BackwardPrefetch, CPUOffload, ShardingStrategy, MixedPrecision, 
        StateDictType, FullStateDictConfig, FullOptimStateDictConfig,
    )
    from accelerate.utils import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    if mixed_precision == "fp16":
        dtype = torch.float16
    elif mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32   
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy = {
            'FULL_SHARD': ShardingStrategy.FULL_SHARD,
            'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
            'NO_SHARD': ShardingStrategy.NO_SHARD,
            'HYBRID_SHARD': ShardingStrategy.HYBRID_SHARD,
            'HYBRID_SHARD_ZERO2': ShardingStrategy._HYBRID_SHARD_ZERO2,
        }[fsdp_cfg.sharding_strategy],
        backward_prefetch = {
            'BACKWARD_PRE': BackwardPrefetch.BACKWARD_PRE,
            'BACKWARD_POST': BackwardPrefetch.BACKWARD_POST,
        }[fsdp_cfg.backward_prefetch],
        mixed_precision_policy = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
        ),
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=fsdp_cfg.min_num_params
        ),
        cpu_offload = CPUOffload(offload_params=fsdp_cfg.cpu_offload),
        state_dict_type = {
            'FULL_STATE_DICT': StateDictType.FULL_STATE_DICT,
            'LOCAL_STATE_DICT': StateDictType.LOCAL_STATE_DICT,
            'SHARDED_STATE_DICT': StateDictType.SHARDED_STATE_DICT
        }[fsdp_cfg.state_dict_type],
        state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        limit_all_gathers = fsdp_cfg.limit_all_gathers,
        use_orig_params = fsdp_cfg.use_orig_params,
        sync_module_states = fsdp_cfg.sync_module_states,
        forward_prefetch = fsdp_cfg.forward_prefetch,
        activation_checkpointing = fsdp_cfg.activation_checkpointing,
    )
    return fsdp_plugin


def freeze_model(model, trainable_modules={}, verbose=False):
    logger.info("Start freeze")
    for name, param in model.named_parameters():
        param.requires_grad = False
        if verbose:
            logger.info("freeze moduel: "+str(name))
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                if verbose:
                    logger.info("unfreeze moduel: "+str(name))
                break
    logger.info("End freeze")
    params_unfreeze = [p.numel() if p.requires_grad == True else 0 for n, p in model.named_parameters()]
    params_freeze = [p.numel() if p.requires_grad == False else 0 for n, p in model.named_parameters()]
    logger.info(f"Unfreeze Module Parameters: {sum(params_unfreeze) / 1e6} M")
    logger.info(f"Freeze Module Parameters: {sum(params_freeze) / 1e6} M")
    return 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(ema_model, 'module'):
        ema_model = ema_model.module
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)



def log_validation(model):
    pass