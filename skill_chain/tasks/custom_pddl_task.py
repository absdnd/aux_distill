from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem
from habitat.tasks.rearrange.multi_task.pddl_task import PddlTask
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_task import NavToInfo
import random
import os.path as osp
import os
import numpy as np
from typing import cast, Optional
@registry.register_task(name="CustomRearrangePddlTask-v0")
class CustomPddlTask(PddlTask):
    def __init__(self, *args, config, **kwargs):
        
        self._recep_to_marker_config = {
            "receptacle_aabb_drawer_left_top_frl_apartment_kitchen_counter": "cab_push_point_7",
            "receptacle_aabb_drawer_middle_top_frl_apartment_kitchen_counter": "cab_push_point_6", 
            "receptacle_aabb_drawer_right_top_frl_apartment_kitchen_counter": "cab_push_point_5",
            "receptacle_aabb_drawer_right_bottom_frl_apartment_kitchen_counter": "cab_push_point_4", 
            "receptacle_aabb_middle_topfrl_apartment_refrigerator":"fridge_push_point"
        }
        
        # use marker is None # 
        self._use_marker = None
        self.seq_skills = config.seq_task_config.use_seq_skills
        self._task_before_grasp = config.seq_task_config.before_grasp_id
        self._task_after_grasp = config.seq_task_config.after_grasp_id
        self._spawn_agent_at_min_distance = config.get('spawn_agent_at_min_distance', False)
        self._min_start_distance = config.get('min_start_distance', None)
        self.should_prevent_drop = config.should_prevent_drop
        self.should_prevent_stop = config.get('should_prevent_stop', False)
        self._assign_marker_to_open_recep = config.get('assign_marker_to_open_recep', False)
        self.task_id = config.task_id
        self.task_type = config.task_spec
        self._no_drop = config.should_prevent_drop
        self._nav_to_info = None
        self._open_ratio = config.open_ratio
        super().__init__(config=config, *args, **kwargs)
    
    def get_sampled(self):
        return [None]

    # Prevent dropping when the object is gripped implement for pick # 
    def _should_prevent_grip(self, action_args):
        return (
            self._sim.grasp_mgr.snap_idx is not None
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] < 0
        )
    
    # Preventing Dropping till Rearrange is Complete # 
    def _should_prevent_drop(self, action_args):
        return (
            self._sim.grasp_mgr.snap_idx is not None
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] <= 0
            and not self.measurements.measures["obj_at_goal"].get_metric()['0']
            and self._no_drop
        )

    def _should_prevent_stop(self, action_args):
        return (
            action_args.get("rearrange_stop", None) is not None
            and action_args["rearrange_stop"] >= 0
            and not self.measurements.measures["obj_at_goal"].get_metric()['0']
            and self.should_prevent_stop
        )
    
    @property
    def success_js_state(self) -> float:
        """
        The success state of the articulated object desired joint.
        """
        if self.task_type == "rearrange_easy" and self._use_marker is not None:
            if self._use_marker == "fridge_push_point":
                return 1.2207963268 * self._open_ratio
            else: 
                return 0.45 * self._open_ratio 
        
        elif self.task_type == "nav_open_cab_pick":
            return 0.45 * self._open_ratio
        
        return self._config.success_state

    # Generate Navigation Start Goal #
    def _generate_nav_start_goal(self, episode, force_idx=None) -> NavToInfo:
        """
        Returns the starting information for a navigate to object task.
        """

        start_hold_obj_idx: Optional[int] = None
        nav_to_pos = self.nav_goal_pos
        def filter_func(start_pos, _):
            return (
                np.linalg.norm(start_pos - nav_to_pos)
                > self._min_start_distance
            )

        (
            articulated_agent_pos,
            articulated_agent_angle,
        ) = self._sim.set_articulated_agent_base_to_random_point(
            filter_func=filter_func
        )

        return articulated_agent_pos, articulated_agent_angle
    
    def _check_any_obj_at_goal(self):
        for _, at_goal in self.measurements.measures["obj_at_goal"].get_metric().items():
            if at_goal:
                return True
        return False

    def _update_to_seq_task_id(self):
        if not self._sim.grasp_mgr.is_grasped:
            self.task_id = self._task_before_grasp
        else: 
            self.task_id = self._task_after_grasp
        
    @property
    def nav_goal_pos(self):
        all_pos = self._sim.get_targets()
        all_obj_pos = self._sim.get_target_objs_start()
        
        if self.task_type == "nav_place":
            return all_pos[1][self._targ_idx]
        
        elif self.task_type == "nav_pick":
            return all_obj_pos[self._targ_idx]
    
        elif self.task_type == "nav_open_cab_pick":
            return all_obj_pos[self._targ_idx]
        
        else: 
            if self._sim.grasp_mgr.is_grasped:
                return all_pos[1][self._targ_idx]
            else:
                return all_obj_pos[self._targ_idx]
    
    def step(self, action, episode): 
        action_args = action["action_args"]
        if self.task_type == "nav_pick":
            if self._should_prevent_grip(action_args):
                action_args["grip_action"] = None

        elif self.task_type == "nav_open_cab_pick":
            if self._should_prevent_grip(action_args):
                action_args["grip_action"] = None
        
        elif self.task_type == "rearrange_easy":
            if self.seq_skills:
                self._update_to_seq_task_id()    
            
            # Equivalent of should prevent grip # 
            if self._should_prevent_drop(action_args):
                action_args["grip_action"] = None
            
            if self._should_prevent_stop(action_args): 
                action_args["rearrange_stop"] = None

        elif self.task_type == "nav_place" and self._no_drop:
            if self._should_prevent_drop(action_args):
                action_args["grip_action"] = None

        obs = super().step(action=action, episode=episode)
        return obs

    # Check if receptacle contains object #
    # Both forms are acceptable # 

    # 1. kitchen_counter_:0000|receptacle_aabb_drawer_middle_top_frl_apartment_kitchen_counter
    # 2. receptacle_aabb_drawer_middle_top_frl_apartment_kitchen_counter

    def is_obj_in_receptacle(self, episode):
        ep_target = [*episode.targets][0]
        receptacle_name = self.preprocess_recep_name(episode.name_to_receptacle[ep_target])
        return (
            receptacle_name in self._recep_to_marker_config 
        )
    
    def is_receptacle_open(self, episode): 
        ep_target = [*episode.targets][0]
        receptacle_name = episode.name_to_receptacle[ep_target] 
        receptacle_name = self.preprocess_recep_name(receptacle_name)
            
        marker_name = self._recep_to_marker_config[receptacle_name]
        joint_pos = self._sim._markers[marker_name].get_targ_js()
        if joint_pos > 0.0:
            return True
        return False

    def preprocess_recep_name(self, receptacle_name):
        if "|" in receptacle_name:
            return receptacle_name.split("|")[1]
        return receptacle_name
 
    @property
    def use_marker_name(self):
        return self._use_marker


    def get_use_marker(self):
        if self._use_marker is not None:
            return self._sim._markers[self._use_marker]
        else: 
            return None

    def reset_marker(self, episode):
        self._use_marker = None
        
        if self.is_obj_in_receptacle(episode):
            if not self.is_receptacle_open(episode) or self._assign_marker_to_open_recep:    
                ep_target = [*episode.targets][0]
                receptacle_name = episode.name_to_receptacle[ep_target]
                receptacle_name = self.preprocess_recep_name(receptacle_name)
                self._use_marker = self._recep_to_marker_config[receptacle_name]
            else: 
                self._use_marker = None
        
    def reset(self, episode: Episode):  
        if self.task_type == "nav_place": 
            sim = self._sim
            sim.grasp_mgr.desnap(force=True)
            super().reset(episode)
            abs_obj_idx = sim.scene_obj_ids[self.abs_targ_idx]
            sim.grasp_mgr.snap_to_obj(abs_obj_idx, force=True) 

        elif self.task_type == "nav_open_cab_pick":
            self.reset_marker(episode)
            super().reset(episode)

        elif self.task_type == "rearrange_easy":
            self.reset_marker(episode)
            super().reset(episode)
            
        else: 
            super().reset(episode)

        if self._spawn_agent_at_min_distance:            
            articulated_agent_start_pos, articulated_agent_start_angle = self._generate_nav_start_goal(episode)
            self._sim.articulated_agent.base_pos = (
                articulated_agent_start_pos
            )
            self._sim.articulated_agent.base_rot = (
                articulated_agent_start_angle
            )

        self.pddl_problem.bind_to_instance(
            self._sim, cast(RearrangeDatasetV0, self._dataset), self, episode
        )

        self._sim.maybe_update_articulated_agent()
        return self._get_observations(episode)        
