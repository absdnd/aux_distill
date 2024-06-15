
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.articulated_object_task import RearrangeOpenFridgeTaskV1, RearrangeOpenDrawerTaskV1, SetArticulatedObjectTask
import numpy as np
from typing import cast, Optional
import magnum as mn
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
import os.path as osp

@registry.register_task(name="AuxRearrangeOpenFridgeTask-v0")
class AuxRearrangeOpenFridgeTaskV1(RearrangeOpenFridgeTaskV1):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            **kwargs,
        )
        self.task_id = self._config.task_id
        self.task_type = "open_fridge"

    @property
    def abs_targ_idx(self):
        return 0

    @property
    def nav_goal_pos(self):
        return np.array([0.,0.,0.])

    def _should_prevent_drop(self, action_args):
        return (
            self._sim.grasp_mgr.is_grasped
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] <= 0
            and not self.measurements.measures["obj_at_goal"].get_metric()['0']
        )

    def step(self, action, episode): 
        obs = super().step(action=action, episode=episode)
        return obs
    
    
@registry.register_task(name="AuxNavRearrangeOpenFridgeTask-v0")
class AuxNavRearrangeOpenFridgeTaskV1(AuxRearrangeOpenFridgeTaskV1):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            **kwargs,
        )
        self.task_id = self._config.task_id
        self._min_start_distance = config.min_start_distance
        self.task_type = "nav_open_fridge"

    # Setting Navigation target so robot is sampled away from it # 
    @property
    def nav_goal_pos(self):
        return self._sim._markers[self._use_marker].get_current_position()
    
    def _sample_robot_start(self, T):
        """
        Returns the starting information for a navigate to object task.
        """

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

        return articulated_agent_angle, articulated_agent_pos

    
        

@registry.register_task(name="AuxRearrangeOpenDrawerTask-v0")
class AuxRearrangeOpenDrawerTaskV1(SetArticulatedObjectTask):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            **kwargs,
        )
        self.task_id = self._config.task_id
        self.task_type = "open_drawer"
        self._use_marker_options = self._config.use_marker_options
        task_spec_path = osp.join(
            config.task_spec_base_path,
            config.task_spec + ".yaml",
        )

        self.pddl_problem = PddlProblem(
            config.pddl_domain_def,
            task_spec_path,
            config,
        )
        self._recep_to_marker_config = {
            "receptacle_aabb_drawer_left_top_frl_apartment_kitchen_counter": "cab_push_point_7",
            "receptacle_aabb_drawer_middle_top_frl_apartment_kitchen_counter": "cab_push_point_6", 
            "receptacle_aabb_drawer_right_top_frl_apartment_kitchen_counter": "cab_push_point_5",
            "receptacle_aabb_drawer_right_bottom_frl_apartment_kitchen_counter": "cab_push_point_4",
        }

    @property
    def abs_targ_idx(self):
        return 0

    @property
    def nav_goal_pos(self):
        return np.array([0.,0.,0.])

    def _get_spawn_region(self):
        return mn.Range2D([0.80, -0.35], [0.95, 0.35])

    def _get_look_pos(self):
        return [0.0, 0.0, 0.0]

    # Generate the start state for the task #
    def _gen_start_state(self):
        drawers = np.zeros((8,))
        return drawers
    
    # Should I prevent the object from dropping? #
    def _should_prevent_drop(self, action_args):
        return (
            self._sim.grasp_mgr.is_grasped
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] <= 0
            and not self.measurements.measures["obj_at_goal"].get_metric()['0']
        )
    
    # Choosing marker based on object position #
    def reset_marker(self, episode: Episode):
        ep_target = [*episode.targets]
        if len(ep_target) == 0:
            self._use_marker = np.random.choice(self._use_marker_options)
            return
        
        receptacle_name = episode.name_to_receptacle[ep_target[0]]
        self._use_marker = self._recep_to_marker_config[receptacle_name]

    def reset(self, episode: Episode):
        self.reset_marker(episode=episode)
        obs = super().reset(episode=episode)
        return obs

    def step(self, action, episode): 
        obs = super().step(action=action, episode=episode)
        return obs

@registry.register_task(name="AuxNavRearrangeOpenDrawerTask-v0")
class AuxRearrangeNavOpenDrawerTaskV1(AuxRearrangeOpenDrawerTaskV1):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(*args, config=config, dataset=dataset, **kwargs)
        self._min_start_distance = config.min_start_distance
        self.task_type = "nav_open_drawer"

    @property
    def nav_goal_pos(self):
        return self._sim._markers[self._use_marker].get_current_position()
        
    # Generate Navigation start goal for that episode # 
    def _sample_robot_start(self, T):
        """
        Returns the starting information for a navigate to object task.
        """
        
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

        return articulated_agent_angle, articulated_agent_pos

    