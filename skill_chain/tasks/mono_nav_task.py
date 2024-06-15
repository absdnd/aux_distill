from habitat.tasks.rearrange.sub_tasks.nav_to_obj_task import DynNavRLEnv
from habitat.core.registry import registry
from skill_chain.tasks.mono_lang_pick_task import PddlMultiTask
from habitat.tasks.rearrange.utils import (
    place_agent_at_dist_from_pos,
    rearrange_logger,
)
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    ExprType, PddlEntity, SimulatorObjectType)
from typing import List, Optional, Tuple
from habitat.articulated_agents.mobile_manipulator import MobileManipulator
import numpy as np
import magnum as mn

def _spawn_robot_at_min_distance(
    target_position: np.ndarray,
    rotation_perturbation_noise: float,
    distance_threshold: float,
    sim,
    num_spawn_attempts: int,
    filter_colliding_states: bool,
    agent: Optional[MobileManipulator] = None,
) -> Tuple[mn.Vector3, float, bool]:
   
    assert (
        distance_threshold > 0.0
    ), f"Distance threshold must be positive, got {distance_threshold=}. You might want `place_agent_at_dist_from_pos` instead."
    if agent is None:
        agent = sim.articulated_agent

    start_rotation = agent.base_rot
    start_position = agent.base_pos

    # Try to place the robot.
    for _ in range(num_spawn_attempts):
        # Place within `distance_threshold` of the object.
        agent.base_pos = sim.pathfinder.get_random_navigable_point_near(
            target_position,
            distance_threshold,
            island_index=sim.largest_island_idx,
        )
        # get_random_navigable_point_near() can return NaNs for start_position.
        # We want to make sure that the generated start_position is valid
        if np.isnan(agent.base_pos).any():
            continue

        # get the horizontal distance (XZ planar projection) to the target position
        hor_disp = agent.base_pos - target_position
        hor_disp[1] = 0
        target_distance = hor_disp.length()

        if target_distance < distance_threshold:
            continue

        # Face the robot towards the object.
        relative_target = target_position - agent.base_pos
        angle_to_object = get_angle_to_pos(relative_target)
        rotation_noise = np.random.normal(0.0, rotation_perturbation_noise)
        agent.base_rot = angle_to_object + rotation_noise

        is_feasible_state = True
        if filter_colliding_states:
            # Make sure the robot is not colliding with anything in this
            # position.
            sim.perform_discrete_collision_detection()
            _, details = rearrange_collision(
                sim,
                False,
                ignore_base=False,
            )

            # Only care about collisions between the robot and scene.
            is_feasible_state = details.robot_scene_colls == 0

        if is_feasible_state:
            propsed_pos = agent.base_pos
            proposed_rot = agent.base_rot
            # found a feasbile state: reset state and return proposed stated
            agent.base_pos = start_position
            agent.base_rot = start_rotation
            return propsed_pos, proposed_rot, False

    # failure to sample a feasbile state: reset state and return initial conditions
    agent.base_pos = start_position
    agent.base_rot = start_rotation
    return start_position, start_rotation, True

    
@registry.register_task(name="MonoNavToObjTask-v0")
class MonoNavToObjTask(DynNavRLEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_id = self._config.task_id
        self._stop_at_goal = self._config.stop_at_goal
        self._use_marker = None
        self.task_type = "mono_nav"

    # Should prevent drop # 
    def _should_prevent_drop(self, action_args):
        return (
            self._sim.grasp_mgr.is_grasped
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] <= 0
        )

    @property
    def use_marker_name(self):
        return self._use_marker

    # What is the success state of the object? # 
    @property
    def success_js_state(self) -> float:
        """
        The success state of the articulated object desired joint.
        """
        return self._config.success_state

    # Get the marker to be used for this task #
    def get_use_marker(self):
        return self._use_marker
    
    def get_sampled(self):
        return [None]

    # Step #
    def step(self, action, episode):  
        action_args = action["action_args"]

        # No arm action during training # 
        if self._config.no_arm_action:        
            action_args["arm_action"][:] = 0.0
        
        if self._should_prevent_drop(action_args):
            action_args["grip_action"] = None
        
        obs = super().step(action=action, episode=episode)
        return obs

# Creating an Object Goal Navigation task using PDDL multi-task #
@registry.register_task(name="MonoObjNavTask-v0")
class MonoObjectNavTask(PddlMultiTask):
    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self.force_set_idx = None
        self._base_angle_noise = self._config.base_angle_noise
        self._spawn_max_dist_to_obj = self._config.spawn_max_dist_to_obj
        self._spawn_min_dist_to_obj = self._config.spawn_min_dist_to_obj
        self._num_spawn_attempts = self._config.num_spawn_attempts
        self._filter_colliding_states = self._config.filter_colliding_states
        self._should_stop_at_goal = self._config.stop_at_goal

    # Sample index to be picked up #
    def _sample_idx(self, sim):
        sel_idx = 0
        return sel_idx

    def get_sampled(self) -> List[PddlEntity]:
        return [self.new_entities[k] for k in self._sampled_names]


    def _get_targ_pos(self, sim):
        scene_pos = sim.get_scene_pos()
        targ_idxs = sim.get_targets()[0]
        return scene_pos[targ_idxs]
    
    @property
    def success_js_state(self) -> float:
        return self._config.success_state

    def _gen_start_pos(self, sim, episode, sel_idx):
        target_positions = self._get_targ_pos(sim)
        targ_pos = target_positions[sel_idx]

        if self._spawn_min_dist_to_obj > 0:
            start_pos, angle_to_obj, was_fail = _spawn_robot_at_min_distance(
                targ_pos,
                self._base_angle_noise,
                self._spawn_min_dist_to_obj,
                sim,
                self._num_spawn_attempts,
                self._filter_colliding_states,
            )
        else: 
            start_pos, angle_to_obj, was_fail = place_agent_at_dist_from_pos(
                targ_pos,
                self._base_angle_noise,
                self._spawn_max_dist_to_obj,
                sim, 
                self._base_angle_noise,
                self._num_spawn_attempts,
                self._filter_colliding_states,
            )
            

        if was_fail:
            rearrange_logger.error(
                f"Episode {episode.episode_id} failed to place robot"
            )

        return start_pos, angle_to_obj

    def reset(self, episode):    
        super().reset(episode)
        self.prev_colls = 0
        sim = self._sim

        sel_idx = self._sample_idx(sim)
        start_pos, start_rot = self._gen_start_pos(sim, episode, sel_idx)

        sim.articulated_agent.base_pos = start_pos
        sim.articulated_agent.base_rot = start_rot

        self._targ_idx = sel_idx

        self._sim.maybe_update_articulated_agent()
        return self._get_observations(episode)
    
    # Take a step in the environment #
    def step(self, action, episode):
        obs = super().step(action, episode)
        return obs
    