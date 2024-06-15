from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.baseline_registry import baseline_registry
from typing import Optional
import torch
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_dones,
    build_rnn_build_seq_info,
)

@baseline_registry.register_storage
class EpisodeNormalizedRolloutStorage(RolloutStorage):
    def __init__(
            self, 
            actor_critic, 
            **kwargs,
    ): 
        super().__init__(actor_critic=actor_critic,**kwargs)

    # Computing returns per environment worker #  
    def compute_returns(self, *args, **kwargs):
        
        super().compute_returns(*args, **kwargs)    
        (done_x, done_y, _) = torch.where(self.buffers["masks"] == 0)

        env_start_idx = {0 for _ in range(self._num_envs)}
        for i, (dx, dy) in enumerate(zip(done_x, done_y)):
            if i + 1 == len(done_y):
                break
            step_slice = slice(env_start_idx[dx.item()], dy.item() + 1)
            buffer_mean = torch.mean(self.buffers[dx, step_slice, :])
            buffer_std = torch.std(self.buffers[dx, step_slice, :])
            self.buffers[dx, step_slice, :] = (
                self.buffers[dx, step_slice, :] - buffer_mean
            )/ (buffer_std + 1e-5)

            env_start_idx[dx.item()] = dy.item() + 1
             
        buffer_mean = self.buffers["returns"].mean(axis=0)
        buffer_std = self.buffers["returns"].std(axis=0)
        self.buffers["returns"] = (
            self.buffers["returns"] - buffer_mean
        )/ (buffer_std + 1e-5)


    

    
    

