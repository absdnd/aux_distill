from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.baseline_registry import baseline_registry
from typing import Optional
import torch
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_dones,
    build_rnn_build_seq_info,
)
from habitat_baselines.utils.timing import g_timer


@baseline_registry.register_storage
class MonoRolloutStorage(RolloutStorage):

    def __init__(
            self, 
            *args,
            **kwargs,
    ): 
        super().__init__(*args, **kwargs)
        self.buffers['active_skill_id'] = torch.zeros(
            self.num_steps + 1, self.num_envs, 1
        )
    
    