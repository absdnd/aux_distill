import inspect
import os.path as osp
import random
from typing import Any, Dict, List

import numpy as np
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    ExprType, PddlEntity, SimulatorObjectType)
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from omegaconf import DictConfig, ListConfig
from habitat.tasks.rearrange.utils import (
    place_agent_at_dist_from_pos,
    rearrange_logger,
)


# import habitat_transformers
from habitat.datasets.rearrange.navmesh_utils import unoccluded_navmesh_snap, snap_point_is_occluded
import os
def get_pddl(pddl_config, domain_spec_path) -> PddlDomain:
    return PddlDomain(
        domain_spec_path,
        pddl_config,
    )



@registry.register_task(name="PddlMultiTask-v0")
class PddlMultiTask(RearrangeTask):
    """
    Task that is specified by a PDDL goal expression and a set of PDDL start
    predicates.
    """

    def __init__(self, *args, config, **kwargs):
        domain_spec_path = osp.join(
            os.getcwd(),
            config.task_spec_base_path,
            "domain.yaml",
        )
        self.pddl = get_pddl(config, domain_spec_path)

        super().__init__(*args, config=config, **kwargs)

        self._start_template = self._config.start_template
        self._goal_template = self._config.goal_template
        self._sample_entities = self._config.sample_entities
        self.fix_position_same_episode = False
        self.fix_target_same_episode = False
        # self.fix_position_same_episode = self._config.fix_position_same_episode
        # self.fix_target_same_episode = self._config.fix_target_same_episode
        self.target_sampling_strategy = self._config.target_sampling_strategy
        self.target_type = self._config.target_type
        self.task_type = self._config.task_spec
        self.task_id = self._config.task_id
        self._last_agent_state = None
        self._should_update_pos = True
        self.agent_height = 1.2
        self.robot_at_threshold = 2
        self.to_exclude_entities = []
        self._use_marker = None

    def _set_articulated_agent_start(self, agent_idx: int) -> None:
        if self._should_update_pos or self._last_agent_state is None:
            super()._set_articulated_agent_start(agent_idx)
        else:
            articulated_agent = self._sim.get_agent_data(
                agent_idx
            ).articulated_agent
            articulated_agent.base_pos = self._last_agent_state[0]
            articulated_agent.base_rot = self._last_agent_state[1]

    def _setup_pddl_entities(self, episode):
        movable_entity_type = self.pddl.expr_types[
            SimulatorObjectType.MOVABLE_ENTITY.value
        ]
        # Register the specific objects in this scene as PDDL entities.
        for obj_name in self.pddl.sim_info.obj_ids:
            asset_name = _strip_instance_id(obj_name)
            asset_type = ExprType(asset_name, movable_entity_type)
            self.pddl.register_type(asset_type)
            self.pddl.register_episode_entity(PddlEntity(obj_name, asset_type))

        robot_entity_type = self.pddl.expr_types[SimulatorObjectType.ROBOT_ENTITY.value]
        for robot_id in self.pddl.sim_info.robot_ids:
            self.pddl.register_episode_entity(PddlEntity(robot_id, robot_entity_type))

    def _load_start_info(self, episode, no_validation=False):
        pddl_entities = self.pddl.all_entities
        self.pddl.bind_to_instance(self._sim, self._dataset, self, episode)
        self._setup_pddl_entities(episode)

        self.new_entities: Dict[str, PddlEntity] = {}
        for entity_name, entity_conds in self._sample_entities.items():
            match_type = self.pddl.expr_types[entity_conds["type"]]
            matches = list(self.pddl.find_entities(match_type))
            # Filter out the extra PDDL entities.
            matches = [match for match in matches if match.expr_type.name not in ["", "any"] and (no_validation or match not in self.to_exclude_entities)]

            if len(matches) == 0:
                raise ValueError(
                    f"Could not find match for {entity_name}: {entity_conds}"
                )

            if self.target_sampling_strategy == "object_type":
                expr_type_names = set([x.expr_type.name for x in matches])
                expr_type_name_rnd = random.choice(list(expr_type_names))
            elif self.target_sampling_strategy == "object_instance":
                expr_type_name_rnd = random.choice(matches).expr_type.name
            else:
                raise ValueError

            if self.target_type == "object_type":
                self.new_entities = {
                    f"obj{i}": ent
                    for i, ent in enumerate(
                        [
                            x
                            for x in matches
                            if x.expr_type.name == expr_type_name_rnd
                        ]
                    )
                }
            elif self.target_type == "object_instance":
                self.new_entities = {
                    f"obj{i}": ent
                    for i, ent in enumerate(
                        [
                            random.choice(
                                [
                                    x
                                    for x in matches
                                    if x.expr_type.name == expr_type_name_rnd
                                ]
                            )
                        ]
                    )
                }
            else:
                raise ValueError

        self.all_obj_pos = [
            self.pddl.sim_info.get_entity_pos(entity)
            for entity in self.new_entities.values()
        ]
        self.all_snapped_obj_pos = [
            unoccluded_navmesh_snap(
                pos, 
                self.agent_height, 
                self._sim.pathfinder, 
                self._sim, 
                island_id=self._sim.largest_island_idx, 
                target_object_id=self.pddl.sim_info.search_for_entity(entity)
            )
            for pos, entity in zip(self.all_obj_pos, self.new_entities.values())
        ]
        to_keep = [
            x is not None and np.linalg.norm(np.asarray((y - x))[[0, 2]]) < self.robot_at_threshold 
            for x, y in zip(self.all_snapped_obj_pos, self.all_obj_pos)
        ]
        if no_validation:
            to_keep = [True] * len(to_keep)
            print(f"Not validating the navigability in episode {episode.episode_id}.")

        if not all(to_keep):
            print(f"Removing {len([x for x in to_keep if not x])} targets out of {len(to_keep)} in episode {episode.episode_id}.")

        if not to_keep or not any(to_keep):
            print(f"Object type {expr_type_name_rnd} is not navigable in episode {episode.episode_id}.")
            self.to_exclude_entities.extend(self.new_entities.values())
            return False
        
        self.all_obj_pos = [x for x, tk in zip(self.all_obj_pos, to_keep) if tk]
        self.all_snapped_obj_pos = [x for x, tk in zip(self.all_snapped_obj_pos, to_keep) if tk]
        self.new_entities = {k: v for (k, v), tk in zip(self.new_entities.items(), to_keep) if tk}
        self._sampled_names = list(self.new_entities.keys())
        self._goal_expr = self._load_goal_preds(episode)
        self._goal_expr, _ = self.pddl.expand_quantifiers(self._goal_expr)
        return len(self.new_entities) > 0
    
    @property
    def use_marker_name(self):
        return self._use_marker
    
    def get_use_marker(self):
        if self._use_marker is not None:
            return self._sim._markers[self._use_marker]
        else: 
            return None
    
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

    def _load_goal_preds(self, episode):
        # Load from the config.
        # goal_d = dict(self._goal_template)
        goal_d = {
            "expr_type": "OR",
            "sub_exprs": [
                "robot_at(obj, robot_0)".replace("obj", name)
                for name in self.new_entities
            ],
        }
        goal_d = _recur_dict_replace(goal_d, self.new_entities)
        return self.pddl.parse_only_logical_expr(goal_d, self.pddl.all_entities)

    def _load_start_preds(self, episode):
        # Load from the config.
        start_preds = self._start_template[:]
        for pred in start_preds:
            for k, entity in self.new_entities.items():
                pred = pred.replace(k, entity.name)
            pred = self.pddl.parse_predicate(pred, self.pddl.all_entities)
            pred.set_state(self.pddl.sim_info)

    def is_goal_satisfied(self):
        return self.pddl.is_expr_true(self._goal_expr)
    
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

    def overwrite_sim_config(self, sim_config: "DictConfig", episode):
        if self.fix_position_same_episode:
            try:
                articulated_agent = self._sim.get_agent_data(
                    0
                ).articulated_agent
                self._last_agent_state = (
                    articulated_agent.base_pos,
                    articulated_agent.base_rot,
                )
            except Exception:
                pass

        return sim_config

    def reset(self, episode):
        self.to_exclude_entities = []
        is_diff_episode = self._episode_id != episode.episode_id

        if not self.fix_position_same_episode or is_diff_episode:
            self._should_update_pos = True
        else:
            self._should_update_pos = False
        self.num_steps = 0

        super().reset(episode, fetch_observations=False)

        if not self.fix_target_same_episode or is_diff_episode:
            for _ in range(10):
                if self._load_start_info(episode):
                    break
            else:
                self._load_start_info(episode, no_validation=True)

            self._load_start_preds(episode)

        self._sim.maybe_update_articulated_agent()
        self.agent_start = self._sim.articulated_agent.base_transformation

        return self._get_observations(episode)

    def get_sampled(self) -> List[PddlEntity]:
        return [self.new_entities[k] for k in self._sampled_names]


def _recur_dict_replace(d: Any, replaces: Dict[str, PddlEntity]) -> Any:
    """
    Replace all string entries in `d` with the replace name to PDDL entity
    mapping in replaces.
    """
    if isinstance(d, ListConfig):
        d = list(d)
    if isinstance(d, DictConfig):
        d = dict(d)

    if isinstance(d, str):
        for name, entity in replaces.items():
            d = d.replace(f"{name}.type", entity.expr_type.name)
            d = d.replace(name, entity.name)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            d[i] = _recur_dict_replace(v, replaces)
    elif isinstance(d, dict):
        for k, v in d.items():
            d[k] = _recur_dict_replace(d[k], replaces)
    return d


def _strip_instance_id(instance_id: str) -> str:
    # Strip off the unique instance ID of the object and only return the asset
    # name.
    return "_".join(instance_id.split("_")[:-1])



@registry.register_task(name="LangPickTaskProx-v0")
class AuxLangPickTask(PddlMultiTask):
    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self.force_set_idx = None
        self._base_angle_noise = self._config.base_angle_noise
        self._spawn_max_dist_to_obj = self._config.spawn_max_dist_to_obj
        self._num_spawn_attempts = self._config.num_spawn_attempts
        self._filter_colliding_states = self._config.filter_colliding_states

    
    def _sample_idx(self, sim):
        sel_idx = 0
        return sel_idx

    def _get_targ_pos(self, sim):
        scene_pos = sim.get_scene_pos()
        targ_idxs = sim.get_targets()[0]
        return scene_pos[targ_idxs]

    def _gen_start_pos(self, sim, episode, sel_idx):
        target_positions = self._get_targ_pos(sim)
        targ_pos = target_positions[sel_idx]

        start_pos, angle_to_obj, was_fail = place_agent_at_dist_from_pos(
            targ_pos,
            self._base_angle_noise,
            self._spawn_max_dist_to_obj,
            sim,
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
    
