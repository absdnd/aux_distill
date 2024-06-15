#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.place_task import RearrangePlaceTaskV1
import numpy as np

@registry.register_task(name="MonoRearrangePlaceTask-v0")
class MonoRearrangePlaceTaskV1(RearrangePlaceTaskV1):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            **kwargs,
        )
        self.use_marker_name = None
        self._no_drop = config.should_prevent_drop
        self.task_id = self._config.task_id
    @property
    def nav_goal_pos(self):
        return np.array([0.,0.,0.])

    def get_use_marker(self):
        return None

    @property
    def success_js_state(self) -> float:
        """
        The success state of the articulated object desired joint.
        """
        return self._config.success_state

    def _should_prevent_drop(self, action_args):
        return (
            self._sim.grasp_mgr.is_grasped
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] <= 0
            and not self.measurements.measures["obj_at_goal"].get_metric()['0']
        )

    def step(self, action, episode): 
        action_args = action["action_args"]
        if self._should_prevent_drop(action_args) and self._no_drop:
            action_args["grip_action"] = None
            
        obs = super().step(action=action, episode=episode)
        return obs
        
@registry.register_task(name="NoDropPlaceTask-v0")
class NoDropPlaceTaskV1(RearrangePlaceTaskV1):
    def _should_prevent_drop(self, action_args):
        return (
            self._sim.grasp_mgr.is_grasped
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] <= 0
            and not self.measurements.measures["obj_at_goal"].get_metric()['0']
        )
    
    def step(self, action, episode): 
        action_args = action["action_args"]
        if self._should_prevent_drop(action_args):
            action_args["grip_action"] = None
        
        obs = super().step(action=action, episode=episode)
        return obs
