from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.rl.ddppo.policy.resnet_policy import (
    PointNavResNetPolicy, 
    PointNavResNetNet, 
    ResNetEncoder,
    ResNetCLIPEncoder,
)
import numpy as np
from habitat_baselines.rl.ddppo.policy import resnet
from gym import spaces
from collections import OrderedDict
import torch 
from torch import nn
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor

from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions

from typing import Optional, List, Dict, Tuple

EPS_PPO = 1e-5
# Critic Head # 


# Using a deep critic head for computing the value-function # 
class DeepCriticHead(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_tasks: int,
        device,
        beta_decay: float = 3e-4,
        use_norm_returns: bool = False,
        use_task_specific_head: bool = False,
        use_pop_art_norm: bool = False,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
        )
        # Setting up number of tasks #
        self._n_tasks = n_tasks
        self._use_type = torch.float32
        self.use_task_specific_head = use_task_specific_head
        self.use_pop_art_norm = use_pop_art_norm
        if self.use_task_specific_head:
            self.fc = nn.Linear(hidden_size, n_tasks)
        else:
            self.fc = nn.Linear(hidden_size, 1)

        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        
        self.mu = torch.zeros((n_tasks, 1), device=device)
        self.nu = torch.ones((n_tasks, 1), device=device)
        self._beta_decay = beta_decay
        self._use_norm_returns = use_norm_returns
        self._env_id_to_task_id = {}

    def _normalize_fc_weights(self, ):
        pass

    def update_stats(self, returns):
        assert returns.dim() == 3
        if not self._use_norm_returns:
            return
        
        avg_returns = returns.mean(0)
        old_mu = self.mu.clone()
        old_nu = self.nu.clone()
        self.mu = (1 - self._beta_decay) * self.mu + self._beta_decay * avg_returns
        self.nu = (1 - self._beta_decay) * self.nu + self._beta_decay * (avg_returns**2)
        
        if self.use_pop_art_norm:
            assert (self.use_task_specific_head, "Pop-art normalization is only supported for task-specific heads")
            old_sigma = torch.sqrt(old_nu - (old_mu**2))
            sigma = torch.sqrt(self.nu - (self.mu**2))
            self.fc.weight.data = ((self.fc.weight.data.t() * old_sigma) / sigma).t()
            self.fc.bias.data = (self.fc.bias.data * old_sigma + old_mu - self.mu)/ sigma

    def post_process_returns(self, returns, task_ids):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(task_ids, op=torch.distributed.ReduceOp.MAX)
        if not self._use_norm_returns:
            return returns
        mu = self.mu[task_ids]
        nu = self.nu[task_ids]
        return (returns - mu) * torch.rsqrt(nu - (mu**2) + EPS_PPO)

    def forward(self, x):
        inner_vf = self.forward_norm(x)
        task_ids = x[:, -self._n_tasks:].argmax(-1)

        if self.use_task_specific_head:
            inner_vf = inner_vf.gather(1, task_ids.unsqueeze(-1))

        assert inner_vf.shape[-1] == 1, "Value function must output a scalar"

        if not self._use_norm_returns:
            return inner_vf

        std = torch.sqrt(self.nu - (self.mu**2))
        mu, std = self.mu[task_ids], std[task_ids]
        return std * inner_vf + mu

    def forward_norm(self, x):
        return self.fc(self.proj(x.to(self._use_type)))



@baseline_registry.register_policy
class AuxMultiTaskPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        fuse_keys: Optional[List[str]] = None,
        use_norm_returns = False,
        **kwargs,
    ):
        """
        Keyword arguments:
        rnn_type: RNN layer type; one of ["GRU", "LSTM"]
        backbone: Visual encoder backbone; one of ["resnet18", "resnet50", "resneXt50", "se_resnet50", "se_resneXt50", "se_resneXt101", "resnet50_clip_avgpool", "resnet50_clip_attnpool"]
        """

        assert backbone in [
            "resnet18",
            "resnet50",
            "resneXt50",
            "se_resnet50",
            "se_resneXt50",
            "se_resneXt101",
            "resnet50_clip_avgpool",
            "resnet50_clip_attnpool",
        ], f"{backbone} backbone is not recognized."

        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        
        super().__init__(
            AuxPointNavResNetNet(
                embed_size = policy_config.get('task_id_embed_dim', None),
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                fuse_keys=fuse_keys,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
                use_mask_sensors=policy_config.get('use_mask_sensors', False),
                use_norm_returns=use_norm_returns,
                use_one_hot_critic=True,
                include_sensor_list=policy_config.get('include_sensor_list', []),

            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

        self.critic = DeepCriticHead(
            hidden_size + observation_space['env_task_id_sensor'].shape[0],
            hidden_size,
            observation_space['env_task_id_sensor'].shape[0],
            torch.device("cuda"),
            beta_decay=policy_config.get('beta_decay', 3e-4),
            use_norm_returns=use_norm_returns,
            use_task_specific_head=policy_config.get('use_task_specific_head', False),
            use_pop_art_norm=policy_config.get('use_pop_art_norm', False),

        )

    # Evaluate Actions of the Policy # 
    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info: Dict[str, torch.Tensor],
    ):
        features, rnn_hidden_states, aux_loss_state = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            rnn_build_seq_info,
        )


        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        batch = dict(
            observations=observations,
            rnn_hidden_states=rnn_hidden_states,
            prev_actions=prev_actions,
            masks=masks,
            action=action,
            rnn_build_seq_info=rnn_build_seq_info,
        )
        aux_loss_res = {
            k: v(aux_loss_state, batch)
            for k, v in self.aux_loss_modules.items()
        }

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            aux_loss_res,
            distribution
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        ignore_names = [
            sensor.uuid
            for sensor in config.habitat_baselines.eval.extra_sim_sensors.values()
        ]
        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k not in ignore_names
                )
            )
        )

        agent_name = None
        if "agent_name" in kwargs:
            agent_name = kwargs["agent_name"]

        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            backbone=config.habitat_baselines.rl.ddppo.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.habitat_baselines.force_blind_policy,
            policy_config=config.habitat_baselines.rl.policy[agent_name],
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            use_norm_returns=config.habitat_baselines.rl.ppo.get('use_norm_returns', False),
            fuse_keys=None,
        )

class AuxPointNavResNetNet(Net):
    
    PRETRAINED_VISUAL_FEATURES_KEY = "visual_features"
    prev_action_embedding: nn.Module
    def __init__(
            self, 
            observation_space,
            embed_size,
            rnn_type, 
            hidden_size, 
            num_recurrent_layers,
            discrete_actions,
            action_space, 
            backbone, 
            resnet_baseplanes, 
            normalize_visual_inputs: bool, 
            fuse_keys, 
            force_blind_policy: bool = False,
            use_mask_sensors: bool = False,
            use_one_hot_critic: bool = False,
            use_norm_returns: bool = False,
            include_sensor_list: list = [],
    ):  
        super().__init__()
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        self._n_prev_action = 32
        self._use_mask_sensors = use_mask_sensors
        self._include_sensor_list = include_sensor_list
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(
                num_actions, self._n_prev_action
            )
        
        
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action  # test
        
        if fuse_keys is None:
            fuse_keys = observation_space.spaces.keys()
            goal_sensor_keys = {
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid,
                ObjectGoalSensor.cls_uuid,
                EpisodicGPSSensor.cls_uuid,
                PointGoalSensor.cls_uuid,
                HeadingSensor.cls_uuid,
                ProximitySensor.cls_uuid,
                EpisodicCompassSensor.cls_uuid,
                ImageGoalSensor.cls_uuid,
                InstanceImageGoalSensor.cls_uuid,
            }
            fuse_keys = [k for k in fuse_keys if k not in goal_sensor_keys]
        

        self._fuse_keys_1d: List[str] = [
            k for k in fuse_keys if len(observation_space.spaces[k].shape) == 1
        ]
       
        if len(self._fuse_keys_1d) != 0:
            for k in self._fuse_keys_1d:
                if k == 'env_task_id_sensor' and embed_size is not None:
                    rnn_input_size += embed_size
                else:
                    rnn_input_size += observation_space.spaces[k].shape[0]

        if embed_size is not None:
            self.embedding_layer = nn.Linear(observation_space['env_task_id_sensor'].shape[0], embed_size)
        else: 
            self.embedding_layer = None

        self._hidden_size = hidden_size
        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        else:
            use_obs_space = spaces.Dict(
                {
                    k: observation_space.spaces[k]
                    for k in fuse_keys
                    if len(observation_space.spaces[k].shape) == 3
                }
            )

        if backbone.startswith("resnet50_clip"):
            self.visual_encoder = ResNetCLIPEncoder(
                observation_space
                if not force_blind_policy
                else spaces.Dict({}),
                pooling="avgpool" if "avgpool" in backbone else "attnpool",
            )
            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Linear(
                        self.visual_encoder.output_shape[0], hidden_size
                    ),
                    nn.ReLU(True),
                )
        else:
            self.visual_encoder = ResNetEncoder(
                use_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )

            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(self.visual_encoder.output_shape), hidden_size
                    ),
                    nn.ReLU(True),
                )

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self._mask_idxs = {}

        # self.critic = MultiLayerCriticHead(hidden_size + observation_space['env_task_id_sensor'].shape[0])

        self._task_id_dim = observation_space['env_task_id_sensor'].shape[0]
        
        self.use_one_hot_critic = use_one_hot_critic
        
        self._embed_size = embed_size

        
        self.train()    

    @property
    def output_size(self):
        if self.use_one_hot_critic:
            return self._hidden_size + self._task_id_dim
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size
    
    @property
    def mask_idxs(self):
        return self._mask_idxs

    
    
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        aux_loss_state = {}
        if not self.is_blind:
            # We CANNOT use observations.get() here because self.visual_encoder(observations)
            # is an expensive operation. Therefore, we need `# noqa: SIM401`
            if (  # noqa: SIM401
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                in observations
            ):
                visual_feats = observations[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ]
            else:
                visual_feats = self.visual_encoder(observations)
            visual_feats = self.visual_fc(visual_feats)
            # aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)


        if len(self._fuse_keys_1d) != 0:
            fuse_states = []
            mask_idxs = {}
            cur_sensor_idx = 0
            for k in self._fuse_keys_1d:
                if k == 'env_task_id_sensor' and self.embedding_layer is not None:
                    obs_embed = self.embedding_layer(observations[k].float())
                    cur_sensor_idx += obs_embed.shape[-1]
                    fuse_states.append(obs_embed)
                else:
                    mask_idxs[k] = [cur_sensor_idx, cur_sensor_idx + observations[k].shape[-1]]
                    cur_sensor_idx += observations[k].shape[-1]
                    fuse_states.append(observations[k])
            
            fuse_states = torch.cat(fuse_states, dim=-1)            

            env_task_ids = torch.argmax(observations['env_task_id_sensor'], dim=-1)            
            task_id_mask = torch.ones_like(fuse_states)
            self._mask_idxs = mask_idxs
            
            ''' 
            Using mask sensors => Removing irrelevant sensors for each task 
            '''

            if self._use_mask_sensors:
                for task_id in torch.unique(env_task_ids):
                    
                    # Removing the articulated sensor from observation space # 
                    if 'is_art_sensor' in self._fuse_keys_1d:
                        task_id_mask[env_task_ids == task_id, mask_idxs['is_art_sensor'][0]:mask_idxs['is_art_sensor'][1]] = 0
                    
                    # Hierarchical Skill sensor # 
                    if 'hier_skill_sensor' in self._fuse_keys_1d:
                        task_id_mask[env_task_ids == task_id, mask_idxs['hier_skill_sensor'][0]:mask_idxs['hier_skill_sensor'][1]] = 0
                    
                    if 'meta_hier_skill_sensor' in self._fuse_keys_1d:
                        task_id_mask[env_task_ids == task_id, mask_idxs['meta_hier_skill_sensor'][0]:mask_idxs['meta_hier_skill_sensor'][1]] = 0

                    # Object Goal Sensor: Pick and Nav + Pikc #
                    if (task_id.item() in [0, 2]):
                        if 'obj_goal_sensor' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_goal_sensor'][0]:mask_idxs['obj_goal_sensor'][1]] = 0
                        if 'goal_to_agent_gps_compass' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['goal_to_agent_gps_compass'][0]:mask_idxs['goal_to_agent_gps_compass'][1]] = 0
                        if 'ee_pos' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['ee_pos'][0]:mask_idxs['ee_pos'][1]] = 0
                        if "marker_rel_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['marker_rel_pos'][0]:mask_idxs['marker_rel_pos'][1]] = 0
                        if 'one_hot_target_sensor' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['one_hot_target_sensor'][0]:mask_idxs['one_hot_target_sensor'][1]] = 0

                    # Place and Nav + Place tasks#
                    elif (task_id.item() in [1, 4]):
                        if 'obj_start_sensor' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_start_sensor'][0]:mask_idxs['obj_start_sensor'][1]] = 0
                        if 'goal_to_agent_gps_compass' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['goal_to_agent_gps_compass'][0]:mask_idxs['goal_to_agent_gps_compass'][1]] = 0
                        if 'ee_pos' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['ee_pos'][0]:mask_idxs['ee_pos'][1]] = 0
                        if "marker_rel_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['marker_rel_pos'][0]:mask_idxs['marker_rel_pos'][1]] = 0
                    
                    # Navigation task # 
                    elif (task_id.item() == 3):
                        if 'obj_start_sensor' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_start_sensor'][0]:mask_idxs['obj_start_sensor'][1]] = 0 
                        if 'obj_goal_sensor' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_goal_sensor'][0]:mask_idxs['obj_goal_sensor'][1]] = 0
                        if 'ee_pos' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['ee_pos'][0]:mask_idxs['ee_pos'][1]] = 0
                        if "marker_rel_pos" in self._fuse_keys_1d: 
                            task_id_mask[env_task_ids == task_id, mask_idxs['marker_rel_pos'][0]:mask_idxs['marker_rel_pos'][1]] = 0
                        if 'one_hot_target_sensor' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['one_hot_target_sensor'][0]:mask_idxs['one_hot_target_sensor'][1]] = 0
                    # If there's a navigation task allow the rearrange-task to access it. # 
                             
                    # Rearrange Task allowing for task to be learnt # 
                    elif (task_id.item() == 5): 
                        if 'marker_rel_pos' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['marker_rel_pos'][0]:mask_idxs['marker_rel_pos'][1]] = 0
                        if 'one_hot_target_sensor' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['one_hot_target_sensor'][0]:mask_idxs['one_hot_target_sensor'][1]] = 0
                            
                    # Pass in marker relative position if it exists.  3
                    elif (task_id.item() in [6, 7, 8, 9]):
                        if "goal_to_agent_gps_compass" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['goal_to_agent_gps_compass'][0]:mask_idxs['goal_to_agent_gps_compass'][1]] = 0
                        if "obj_goal_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_goal_sensor'][0]:mask_idxs['obj_goal_sensor'][1]] = 0
                        if 'one_hot_target_sensor' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['one_hot_target_sensor'][0]:mask_idxs['one_hot_target_sensor'][1]] = 0
                        # Allowing object start sensor to be used for the rearrange task #
                        # if "obj_start_sensor" in self._fuse_keys_1d:
                            # task_id_mask[env_task_ids == task_id, mask_idxs['obj_start_sensor'][0]:mask_idxs['obj_start_sensor'][1]] = 0

                    # Fridge Pick skills # 
                    elif (task_id.item() in [10, 11]):
                        if "goal_to_agent_gps_compass" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['goal_to_agent_gps_compass'][0]:mask_idxs['goal_to_agent_gps_compass'][1]] = 0
                        if "obj_goal_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_goal_sensor'][0]:mask_idxs['obj_goal_sensor'][1]] = 0
                        if "ee_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['ee_pos'][0]:mask_idxs['ee_pos'][1]] = 0
                        if "marker_rel_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['marker_rel_pos'][0]:mask_idxs['marker_rel_pos'][1]] = 0
                        if 'one_hot_target_sensor' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['one_hot_target_sensor'][0]:mask_idxs['one_hot_target_sensor'][1]] = 0

                    # Nav Open-Cabinet + Pick object # 
                    elif task_id.item() == 12:
                        if "goal_to_agent_gps_compass" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['goal_to_agent_gps_compass'][0]:mask_idxs['goal_to_agent_gps_compass'][1]] = 0
                        
                        if "obj_goal_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_goal_sensor'][0]:mask_idxs['obj_goal_sensor'][1]] = 0
                         
                        if "marker_rel_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['marker_rel_pos'][0]:mask_idxs['marker_rel_pos'][1]] = 0
                        
                        if 'one_hot_target_sensor' in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['one_hot_target_sensor'][0]:mask_idxs['one_hot_target_sensor'][1]] = 0

                    # Language Pick tasks #                
                    elif task_id.item() == 13:
                        if "obj_goal_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_goal_sensor'][0]:mask_idxs['obj_goal_sensor'][1]] = 0
                        if "goal_to_agent_gps_compass" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['goal_to_agent_gps_compass'][0]:mask_idxs['goal_to_agent_gps_compass'][1]] = 0
                        if "obj_start_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_start_sensor'][0]:mask_idxs['obj_start_sensor'][1]] = 0
                        if "ee_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['ee_pos'][0]:mask_idxs['ee_pos'][1]] = 0
                        if "marker_rel_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['marker_rel_pos'][0]:mask_idxs['marker_rel_pos'][1]] = 0

                    # Language Place tasks #
                    elif task_id.item() == 14:
                        if "obj_goal_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_goal_sensor'][0]:mask_idxs['obj_goal_sensor'][1]] = 0
                        if "goal_to_agent_gps_compass" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['goal_to_agent_gps_compass'][0]:mask_idxs['goal_to_agent_gps_compass'][1]] = 0
                        if "obj_start_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_start_sensor'][0]:mask_idxs['obj_start_sensor'][1]] = 0
                        if "ee_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['ee_pos'][0]:mask_idxs['ee_pos'][1]] = 0
                        if "marker_rel_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['marker_rel_pos'][0]:mask_idxs['marker_rel_pos'][1]] = 0

                
                    elif task_id.item() == 15:
                        if "obj_goal_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_goal_sensor'][0]:mask_idxs['obj_goal_sensor'][1]] = 0
                        if "goal_to_agent_gps_compass" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['goal_to_agent_gps_compass'][0]:mask_idxs['goal_to_agent_gps_compass'][1]] = 0
                        if "obj_start_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_start_sensor'][0]:mask_idxs['obj_start_sensor'][1]] = 0
                        if "ee_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['ee_pos'][0]:mask_idxs['ee_pos'][1]] = 0
                        if "marker_rel_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['marker_rel_pos'][0]:mask_idxs['marker_rel_pos'][1]] = 0
                        
                    elif task_id.item() == 16:
                        if "obj_goal_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_goal_sensor'][0]:mask_idxs['obj_goal_sensor'][1]] = 0
                        if "goal_to_agent_gps_compass" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['goal_to_agent_gps_compass'][0]:mask_idxs['goal_to_agent_gps_compass'][1]] = 0
                        if "obj_start_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_start_sensor'][0]:mask_idxs['obj_start_sensor'][1]] = 0
                        if "ee_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['ee_pos'][0]:mask_idxs['ee_pos'][1]] = 0
                        if "marker_rel_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['marker_rel_pos'][0]:mask_idxs['marker_rel_pos'][1]] = 0
                    
                    # If task Id is 17. # 
                    elif task_id.item() == 17:
                        if "obj_goal_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_goal_sensor'][0]:mask_idxs['obj_goal_sensor'][1]] = 0
                        if "obj_start_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_start_sensor'][0]:mask_idxs['obj_start_sensor'][1]] = 0
                        if "ee_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['ee_pos'][0]:mask_idxs['ee_pos'][1]] = 0
                        if "marker_rel_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['marker_rel_pos'][0]:mask_idxs['marker_rel_pos'][1]] = 0
                        if "goal_to_agent_gps_compass" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['goal_to_agent_gps_compass'][0]:mask_idxs['goal_to_agent_gps_compass'][1]] = 0
                    
                    elif task_id.item() == 18:
                        if "obj_goal_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_goal_sensor'][0]:mask_idxs['obj_goal_sensor'][1]] = 0
                        if "obj_start_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['obj_start_sensor'][0]:mask_idxs['obj_start_sensor'][1]] = 0
                        if "ee_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['ee_pos'][0]:mask_idxs['ee_pos'][1]] = 0
                        if "marker_rel_pos" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['marker_rel_pos'][0]:mask_idxs['marker_rel_pos'][1]] = 0
                        if "one_hot_target_sensor" in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs['one_hot_target_sensor'][0]:mask_idxs['one_hot_target_sensor'][1]] = 0

                    for sensor_name in self._include_sensor_list: 
                        if sensor_name in self._fuse_keys_1d:
                            task_id_mask[env_task_ids == task_id, mask_idxs[sensor_name][0]:mask_idxs[sensor_name][1]] = 1.0
                fuse_states = fuse_states * task_id_mask
            x.append(fuse_states.float())


        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            # The mask means the previous action will be zero, an extra dummy action
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(
                masks * prev_actions.float()
            )

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        

        if self.use_one_hot_critic:
            out = torch.cat((out, observations['env_task_id_sensor'].float()), dim=-1)
        
        # aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state

        