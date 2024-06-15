from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.registry import registry

from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface
from habitat.core.embodied_task import Measure
from habitat.tasks.nav.nav import PointGoalSensor
from gym import spaces
from habitat.tasks.rearrange.rearrange_sensors import (
    EndEffectorToObjectDistance,
    EndEffectorToRestDistance,
    ForceTerminate,
    RearrangeReward,
    RobotForce,
    MultiObjSensor, 
)
from skill_chain.sensors.mono_nav_to_obj_sensors import MonoNavToObjSuccess
from habitat.tasks.rearrange.utils import batch_transform_point
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import NavToObjSuccess, RotDistToGoal, DistToGoal
import numpy as np
from habitat.tasks.rearrange.utils import rearrange_logger
from typing import List
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import NavToObjSuccess, RotDistToGoal, DistToGoal, NavToPosSucc, DoesWantTerminate
from habitat.tasks.rearrange.sub_tasks.articulated_object_sensors import (
    ArtObjAtDesiredState, 
    ArtObjState,
    ArtObjSuccess,
    EndEffectorDistToMarker,

)
import copy
from habitat.tasks.rearrange.multi_task.pddl_sensors import MoveObjectsReward
import torch
from habitat.tasks.rearrange.rearrange_sensors import (
    DoesWantTerminate,
    EndEffectorToObjectDistance,
    ObjectToGoalDistance,
    RearrangeReward,
)
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import LogicalExpr
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate

from typing import List, Dict, Tuple
# Meta Skill ID I'm distilling from #  
def assign_meta_skill_id(
    allowed_skills, 
    is_art_task, 
    is_holding,
    prev_art_success
):
    if set(allowed_skills) == set(["nav_pick", "nav_place", "nav_open_cabinet", "nav_open_cabinet_pick", "nav_pick_nav_place"]):
        if is_art_task == 2.0:
            if not is_holding:
                return np.array([9.0])
            else:
                return np.array([-1.0])
        else: 
            return np.array([-1.0])
    
    elif set(allowed_skills) == set(["nav_pick", "nav_place", "nav_open_fridge", "nav_open_cabinet_pick", "nav_open_cabinet", "open_fridge_pick", "nav_pick_nav_place"]):
        if not is_holding:
            if is_art_task == 1.0: 
                return np.array([-1.0])
            elif is_art_task == 2.0 and (not prev_art_success):
                return np.array([9.0])
            else: 
                return np.array([-1.0])
        else:
            return np.array([-1.0])

    else:
        return np.array([-1.0])


def assign_distill_skill_id(
        allowed_skills, 
        is_art_task, 
        prev_art_success, 
        prev_nav_success, 
        prev_nav_to_pos_success, 
        is_holding
    ):
    if set(allowed_skills) == set(["nav", "pick", "nav_place", "open_fridge", "open_cabinet", "nav_pick_nav_place"]):
    
        if prev_nav_success["hold_0.0"] and not is_holding:
            if is_art_task == 1.0 and not prev_art_success:
                return np.array([6.0])
                
            elif is_art_task == 2.0 and not prev_art_success:
                return np.array([7.0])
                
            else:
                return np.array([0.0])

        elif not prev_nav_success['hold_0.0'] and not is_holding:
            return np.array([3.0])

        elif is_holding:
            return np.array([4.0])

        else: 
            raise ValueError("Unrecognized State of the Agent")


    elif set(allowed_skills) == set(['nav_pick','nav_place','nav_open_cabinet','nav_open_fridge','nav_pick_nav_place']):

        # Object is not being held by the robot # 
        if not is_holding: 
            if is_art_task == 1.0 and (not prev_art_success):
                return np.array([8.0])
            elif is_art_task == 2.0 and (not prev_art_success):
                return np.array([9.0])
            elif is_art_task == 0.0:
                return np.array([2.0])
            else: 
                return np.array([-1.0])
        else:
            return np.array([4.0])
    
    elif set(allowed_skills) == set(['nav_pick','nav_place','nav_open_cabinet','nav_open_fridge','open_fridge_pick','nav_pick_nav_place']):
        if not is_holding: 
            if is_art_task == 1.0:
                if not prev_art_success:
                    return np.array([8.0])
                else: 
                    return np.array([10.0])                
            elif is_art_task == 2.0:
                if not prev_art_success:
                    return np.array([9.0])
                else: 
                    return np.array([-1.0])
            elif is_art_task == 0.0:
                return np.array([2.0])
            else: 
                return np.array([-1.0])
        else:
            return np.array([4.0])


    
    # Distilling from the open-fridge pick skill here. #  
    elif set(allowed_skills) == set(['nav_pick','nav_place','nav_open_cabinet','nav_pick_nav_place', 'open_cabinet_pick']) or \
        set(allowed_skills) == set(['nav_pick','nav_place','nav_open_fridge','nav_pick_nav_place', 'open_fridge_pick']) or \
        set(allowed_skills) == set(['nav_pick','nav_place','nav_open_fridge','nav_open_cabinet','nav_pick_nav_place','open_cabinet_pick','open_fridge_pick']):
        if not is_holding: 
            if is_art_task == 1.0:
                if not prev_art_success:
                    return np.array([8.0])
                else: 
                    return np.array([10.0])                
            elif is_art_task == 2.0:
                if not prev_art_success:
                    return np.array([9.0])
                else: 
                    return np.array([11.0])
            elif is_art_task == 0.0:
                return np.array([2.0])
            else: 
                return np.array([-1.0])
        else:
            return np.array([4.0])

    # Nav + Pick, Nav + Place and Rearrange #         
    elif set(allowed_skills) == set(['nav_pick', 'nav_place', 'nav_pick_nav_place']):
        if not is_holding: 
            return np.array([2.0])
        
        else:
            return np.array([4.0])

    # Nav, Pick, Place, Open-Fridge, Open-Cabinet, Rearrange # 
    elif set(allowed_skills) == set(['nav', 'pick','place','open_fridge','open_cabinet','nav_pick_nav_place']):            
        if not is_holding:
            if prev_nav_success["hold_0.0"]:
                if is_art_task == 1.0 and not prev_art_success:
                    return np.array([6.0])
                elif is_art_task == 2.0 and not prev_art_success:
                    return np.array([7.0])
                else:
                    return np.array([0.0])
            else: 
                return np.array([3.0])
    
        else: 
            if prev_nav_success["hold_1.0"]:
                return np.array([4.0])

            else:
                return np.array([3.0]) 

    # Computing the Articulated Task using Nav, Pick, Place, Open-Fridge # 
    elif set(allowed_skills) == set(['nav', 'pick', 'place', 'open_fridge', 'nav_pick_nav_place']) or \
        set(allowed_skills) == set(['nav', 'pick', 'place', 'open_cabinet', 'nav_pick_nav_place']):
        if not is_holding:
            if prev_nav_success["hold_0.0"]:
                if is_art_task == 1.0 and not prev_art_success:
                    return np.array([6.0])
                elif is_art_task == 2.0 and not prev_art_success:
                    return np.array([7.0])
                else:
                    return np.array([0.0])
            else: 
                return np.array([3.0])
    
        else: 
            if prev_nav_success["hold_1.0"]:
                return np.array([4.0])

            else:
                return np.array([3.0])

    # Distill with Open-Cabinet till Fridge is Open # 
    elif set(allowed_skills) == set(['nav_open_fridge', 'pick', 'nav_place', 'nav_pick_nav_place']) or \
        set(allowed_skills) == set(['nav_open_cabinet', 'pick', 'nav_place', 'nav_pick_nav_place']):
        if not is_holding:
            if is_art_task == 1.0 and (not prev_art_success):
                return np.array([8.0])
            elif is_art_task == 2.0 and (not prev_art_success):
                return np.array([9.0])
            else: 
                assert (is_art_task != 0.0, "Skill Combination only supports Articulated Tasks")
                return np.array([0.0])
        else: 
            return np.array([4.0])

    elif set(allowed_skills) == set(['nav_open_cabinet', 'nav_pick', 'nav_place', 'nav_pick_nav_place']) or \
        set(allowed_skills) == set(['nav_open_fridge', 'nav_pick', 'nav_place', 'nav_pick_nav_place']):
        if not is_holding:
            if is_art_task == 1.0 and not (prev_art_success):
                return np.array([8.0])
            elif is_art_task == 2.0 and not (prev_art_success):
                return np.array([9.0])
            elif is_art_task == 0.0:
                return np.array([2.0])
            else: 
                return np.array([-1.0])
        else: 
            return np.array([4.0])
        
    ########## Meta Distillation Based on the Articulated Task ##########
    elif set(allowed_skills) == set(['nav_open_cabinet_pick', 'nav_open_cabinet', 'nav_pick', 'nav_place', 'nav_pick_nav_place']):
        if not is_holding:
            if is_art_task == 1.0:
                return np.array([-1.0])
            elif is_art_task == 2.0:
                return np.array([12.0])
            elif is_art_task == 0.0:
                return np.array([2.0])
            else: 
                return np.array([-1.0])
        else: 
            return np.array([4.0])
    
    elif set(allowed_skills) == set(['nav_open_cabinet', 'nav_open_cabinet_pick', 'nav_open_fridge', 'open_fridge_pick', 'nav_pick', 'nav_place', 'nav_pick_nav_place']):
        if not is_holding:
            if is_art_task == 1.0: 
                if not (prev_art_success):
                    return np.array([8.0])
                else: 
                    return np.array([10.0])
            elif is_art_task == 2.0:
                return np.array([12.0])
            else:
                return np.array([2.0])
        else:
            return np.array([4.0])
        
    elif set(allowed_skills) == set(['nav_pick', 'nav_place', 'nav_open_cabinet', 'nav_open_fridge', 'open_fridge_pick']):
        return np.array([-1.0])

    # Is the placing skill even required? #         
    elif set(allowed_skills) == set(['nav_open_cabinet', 'nav_open_fridge', 'open_fridge_pick', 'nav_pick',  'nav_pick_nav_place']):
        if not is_holding:
            if is_art_task == 1.0:
                if not (prev_art_success):
                    return np.array([8.0])
                else: 
                    return np.array([10.0])
            elif is_art_task == 2.0:
                if not (prev_art_success):
                    return np.array([9.0])
                else:
                    return np.array([-1.0])
            else:
                return np.array([2.0])
        else:
            return np.array([-1.0])
    
    ############# Ablation Study for the Articulated Task #############
    elif set(allowed_skills) == set(['nav_pick', 'nav_place', 'open_fridge_pick', 'nav_pick_nav_place']):
        if not is_holding:
            if is_art_task == 1.0:
                if not (prev_art_success):
                    return np.array([-1.0])
                else: 
                    return np.array([10.0])
            elif is_art_task == 2.0:
                return np.array([-1.0])
            else:
                return np.array([2.0])
        else:
            return np.array([4.0])

    elif set(allowed_skills) == set(['nav_place', 'open_fridge_pick', 'nav_open_fridge', 'nav_pick_nav_place']):
        if not is_holding:
            if is_art_task == 1.0:
                if not (prev_art_success):
                    return np.array([8.0])
                else: 
                    return np.array([10.0])
            elif is_art_task == 2.0:
                return np.array([-1.0])
            else:
                return np.array([-1.0])
        else:
            return np.array([4.0]) 
    
    elif set(allowed_skills) == set(['nav_pick', 'nav_open_fridge', 'open_fridge_pick, nav_pick_nav_place']):
        if not is_holding:
            if is_art_task == 1.0:
                if not (prev_art_success):
                    return np.array([8.0])
                else: 
                    return np.array([10.0])
            elif is_art_task == 2.0:
                return np.array([-1.0])
            else:
                return np.array([2.0])
        else:
            return np.array([-1.0])

    elif set(allowed_skills) == set(['nav_pick', 'nav_place', 'nav_open_fridge', 'open_fridge_pick']):
        return np.array([-1.0])
        
    elif set(allowed_skills) == set(['pick', 'lang_pick']) or \
        set(allowed_skills) == set(['pick', 'lang_pick_prox']):
        return np.array([0.0])
    
    # Finding relevant skill with object goal navigation # 
    elif set(allowed_skills) == set(['obj_nav','obj_nav_last_mile','obj_nav_find_obj']):
        if (not prev_nav_to_pos_success["hold_0.0"]) \
                and (not prev_nav_to_pos_success["hold_1.0"]):
            return np.array([15.0])       
        else: 
            return np.array([16.0])
        
    # Return object navigation or navigation target # 
    elif set(allowed_skills) == set(['obj_nav', 'nav']):
        return np.array([3.0])

    elif set(allowed_skills) == set(['coord_nav', 'obj_nav']):
        return np.array([18.0])
    
    elif set(allowed_skills) == set(['nav_pick', 'lang_pick']):
        return np.array([2.0])

    elif (len(allowed_skills) == 1) or (len(allowed_skills) == 0):
        return np.array([-1.0])
    elif set(allowed_skills) == set(['nav_pick', 'nav_place', 'open_cabinet', 'nav_pick_nav_place']):
        return np.array([-1.0])
    else: 
        raise ValueError(f"Unrecognized Skill Combination {allowed_skills}")

    ### No Placing skill #### 
    
@registry.register_sensor
class OneHotTargetSensor(Sensor):
    def __init__(self, *args, task, **kwargs):
        self._task = task
        # TODO: Hard-coded for the ycb objects. Change to work with any object
        # set.
        self._all_objs = [
            "002_master_chef_can",
            "003_cracker_box",
            "004_sugar_box",
            "005_tomato_soup_can",
            "007_tuna_fish_can",
            "008_pudding_box",
            "009_gelatin_box",
            "010_potted_meat_can",
            "011_banana",
            "012_strawberry",
            "013_apple",
            "014_lemon",
            "015_peach",
            "016_pear",
            "017_orange",
            "018_plum",
            "021_bleach_cleanser",
            "024_bowl",
            "025_mug",
            "026_sponge",
        ]
        self._n_cls = len(self._all_objs)

        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return "one_hot_target_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(self._n_cls,), low=0, high=1, dtype=np.float32
        )

    def get_observation(self, *args, **kwargs):
        cur_target = self._task.get_sampled()[0]
        obs = np.zeros((self._n_cls,))

        if cur_target is not None:
            # For receptacles the name will not be a class but the name directly.
            use_name = cur_target.expr_type.name
            if cur_target.name in self._all_objs:
                use_name = cur_target.name

            if use_name not in self._all_objs:
                raise ValueError(
                    f"Object not found given {use_name}, {cur_target}, {self._task.get_sampled()}"
                )
            set_i = self._all_objs.index(use_name)

            if use_name in self._all_objs:
                set_i = self._all_objs.index(use_name)
                obs[set_i] = 1.0

        return obs


@registry.register_sensor
class EnvTaskSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "env_task_id_sensor"

    def __init__(self, sim, config, task, *args, **kwargs):
        self.config = config
        super().__init__(config=config)
        self._task = task
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return EnvTaskSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    # Get observation space for the task #
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(self.config.max_skills,), 
            low=0,
            high=1,
            dtype=np.float32,
        )
    
    def get_observation(self, observations, episode, *args, **kwargs):
        one_hot_task_id = np.eye(self.config.max_skills)[self._task.task_id]
        return one_hot_task_id.reshape((self.config.max_skills,))


@registry.register_measure
class EnvTaskMeasure(Measure):
    cls_uuid: str = "env_task_measure"
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._allowed_skills = config.get('allowed_skills', [])
        # assert len(self._allowed_skills) > 0, "No allowed skills specified"
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EnvTaskMeasure.cls_uuid
    
    @property
    def skill_names(self):
        return self._allowed_skills

    # Custom name for each task # 
    def reset_metric(self, task, *args, **kwargs):
        self._metric = -1
    
    def update_metric(self, task,  *args, **kwargs):
        self._metric = task.task_id    


@registry.register_sensor
class MonoTargetStartSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    Relative position from the End Effector to Objects in the Scene
    """

    cls_uuid: str = "obj_start_sensor"
    def get_observation(self, observations, episode, *args, **kwargs):
        global_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        T_inv = global_T.inverted()
        pos = self._sim.get_target_objs_start()
        if len(pos) > 0:
            return batch_transform_point(pos, T_inv, np.float32).reshape(-1)
        else: 
            return np.array([0.0, 0.0, 0.0])
    
    def _get_observation_space(self, *args, **kwargs):
        n_targets = max(1, self._task.get_n_targets())
        return spaces.Box(
            shape=(n_targets * 3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )


@registry.register_measure
class MonoArtSuccessPos(Measure):
    cls_uuid: str = "art_success_pos"
    def __init__(self, sim, config, task, *args, **kwargs):
        self._sim = sim
        self._task = task
        self._config = config
        super().__init__(**kwargs)
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoArtSuccessPos.cls_uuid

    def reset_metric(self, task, *args, **kwargs):
        self._metric = -1
    
    def update_metric(self, task,  *args, **kwargs):
        if task.use_marker_name is None:
            self._metric = -1
        else:
            self._metric = task.success_js_state

    

# class MonoMoveObjectsReward(MoveObjectsReward):
#     cls_uuid: str = "move_obj_reward"
    
#     @staticmethod
#     def _get_uuid(*args, **kwargs):
#         return MonoMoveObjectsReward.cls_uuid

#     def update_target_object(self):
#         """
#         The agent just finished one rearrangement stage so it's time to
#         update the target object for the next stage.
#         """
#         # Get the next target object
#         idxs, _ = self._sim.get_targets()
#         targ_obj_idx = idxs[self._cur_rearrange_stage]

#         # Get the target object's absolute index
#         self.abs_targ_obj_idx = self._sim.scene_obj_ids[targ_obj_idx]


"""
Mono Object Goal Sensor, using articulated agent interface 
"""
@registry.register_sensor
class MonoGoalSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    Relative to the end effector
    """

    cls_uuid: str = "obj_goal_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        global_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        T_inv = global_T.inverted()

        _, pos = self._sim.get_targets()
        if len(pos) > 0:
            return batch_transform_point(pos, T_inv, np.float32).reshape(-1)
        else: 
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _get_observation_space(self, *args, **kwargs):
        n_targets = max(1, self._task.get_n_targets())
        return spaces.Box(
            shape=(n_targets * 3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )


@registry.register_measure
class IsArticulatedTaskMeasure(Measure):
    cls_uuid: str = "is_art_measure"
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        super().__init__(**kwargs)
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return IsArticulatedTaskMeasure.cls_uuid
    
    def reset_metric(self, task, *args, **kwargs):
        self._metric = -1.0
        self.update_metric(task, *args, **kwargs)
    
    # Update Metric based on Task Specification # 
    def update_metric(self, task, *args, **kwargs):
        if task.use_marker_name is None:
            self._metric = 0.0
        elif task.use_marker_name == "fridge_push_point":
            self._metric = 1.0
        else:
            self._metric = 2.0

@registry.register_measure
class AllRankIsArticulatedTaskMeasure(Measure):
    cls_uuid: str = "all_rank_is_art_measure"
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        super().__init__(**kwargs)
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return AllRankIsArticulatedTaskMeasure.cls_uuid
    
    def reset_metric(self, task, *args, **kwargs):
        self._metric = -1.0
        # self.update_metric(task, *args, **kwargs)
    
    def update_metric(self, task, *args, **kwargs):
        if task.use_marker_name is None:
            self._metric = 0.0
        elif task.use_marker_name == "fridge_push_point":
            self._metric = 1.0
        else:
            self._metric = 2.0



@registry.register_sensor
class IsArticulatedSensor(Sensor):
    cls_uuid: str = "is_art_sensor"
    def __init__(self, sim, config, task, *args, **kwargs):
        self._sim = sim
        self._task = task
        super().__init__(**kwargs)
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return IsArticulatedSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(1,), 
            low=0,
            high=2,
            dtype=np.float32,
        )
    
    def get_observation(self, observations, episode, *args, **kwargs):
        if self._task.use_marker_name is None:
            return np.array([0.0])
        elif self._task.use_marker_name == "fridge_push_point":
            return np.array([1.0])
        else:
            return np.array([2.0])

@registry.register_sensor
class MonoMetaHierSkillSensor(Sensor):
    cls_uuid: str = "meta_hier_skill_sensor"
    def __init__(self, sim, config, task, *args, **kwargs):
        self._sim = sim
        self._task = task
        self._config = config
        self._prev_nav_success = {
            "hold_0.0": False, 
            "hold_1.0": False
        } 

        self._prev_nav_to_pos_success = {
            "hold_0.0": False, 
            "hold_1.0": False
        }
        self._prev_art_success = False
        super().__init__(**kwargs)
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoMetaHierSkillSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(1,), 
            low=-1,
            high=20,
            dtype=np.float32,
        )
    
    def get_observation(self, observations, episode, *args, **kwargs):

        is_holding = bool(self._sim.grasp_mgr._snapped_obj_id is not None)        
        nav_to_obj_succ_cls = self._task.measurements.measures.get(MonoNavToObjSuccess.cls_uuid, None)
        nav_to_pos_succ_cls = self._task.measurements.measures.get(NavToPosSucc.cls_uuid, None)
        art_success_cls = self._task.measurements.measures[self._config.art_success_id]
        is_art_task_cls = self._task.measurements.measures.get(AllRankIsArticulatedTaskMeasure.cls_uuid, None)

        nav_to_obj_succ = None
        art_obj_success = None
        is_art_task = -1.0


        ### First Step of the Episode : Reset all the metrics ###
        if nav_to_obj_succ_cls is not None: 
            nav_to_obj_succ = nav_to_obj_succ_cls.get_metric()
        
        if nav_to_pos_succ_cls is not None:
            nav_to_pos_succ = nav_to_pos_succ_cls.get_metric()

        if art_success_cls is not None:
            art_obj_success = art_success_cls.get_metric()

        if is_art_task_cls is not None:
            is_art_task = is_art_task_cls.get_metric()

        if nav_to_obj_succ is None or is_art_task == -1.0:
            # assert len(observations.keys()) == 1, "Only the head depth should be assigned at the beginning of the episode "
            self._prev_nav_success = {k: False for k in self._prev_nav_success.keys()}
            self._prev_nav_to_pos_success = {k: False for k in self._prev_nav_to_pos_success.keys()}
            self._prev_art_success = False
            return np.array([-1.0])


        self._prev_nav_success["hold_{:.1f}".format(is_holding)] = nav_to_obj_succ or self._prev_nav_success["hold_{:.1f}".format(is_holding)]
        if self._task.use_marker_name is not None:
            self._prev_art_success = art_obj_success or self._prev_art_success

        assert isinstance(self._prev_art_success, bool), "Articulated Success is not a boolean"
        assert isinstance(self._prev_nav_success["hold_0.0"], (bool, np.bool_)), "Navigation Success is not a boolean"
        assert isinstance(self._prev_nav_success["hold_1.0"], (bool, np.bool_)), "Navigation Success is not a boolean"
        
        # Assign distillation skill id # 
        dist_skill_id = assign_meta_skill_id(
            allowed_skills = self._config.allowed_skills, 
            is_art_task=is_art_task, 
            is_holding=is_holding, 
            prev_art_success=self._prev_art_success
        )
        
        return dist_skill_id
@registry.register_measure
class MonoTaskStageSteps(Measure):
    cls_uuid: str = "task_stage_steps"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._allowed_skills = config.allowed_skills
        self._allowed_skill_ids = config.allowed_skill_ids
        self._max_episode_steps = config.max_episode_steps
        
        self._allowed_skills.append("open_cab_pick")
        self._allowed_skill_ids.append(-1)
        self._max_episode_steps.append(300)

        self._skill_id_to_name = {
            self._allowed_skill_ids[i]: self._allowed_skills[i] for i in range(len(self._allowed_skills))
        } 

        self._skill_id_to_max_steps = {
            self._allowed_skill_ids[i]: config.max_episode_steps[i] for i in range(len(self._allowed_skills))
        }

        
        self._max_steps = config.max_episode_steps
        super().__init__(**kwargs)
    
    def _get_uuid(self, *args, **kwargs):
        return MonoTaskStageSteps.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = {}
        for skill_name in self._allowed_skills:
            self._metric[skill_name] = 0

        self.update_metric(*args, episode=episode, task=task, observations=observations, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        assert ('hier_skill_sensor' in observations.keys()), "Hierarchical Skill Sensor not found"
        hier_skill_id = observations['hier_skill_sensor'][0]
        hier_skill_name = self._skill_id_to_name[hier_skill_id]
        self._metric[hier_skill_name] += 1

        if task.task_type=="rearrange_easy" and (self._metric[hier_skill_name] > self._skill_id_to_max_steps[hier_skill_id]):
            if self._config.force_early_end:
                task.should_end = True
        
    

@registry.register_sensor
class MonoHierSkillSensor(Sensor):
    cls_uuid: str = "hier_skill_sensor"
    def __init__(self, sim, config, task, *args, **kwargs):
        self._sim = sim
        self._task = task
        self._config = config
        self._prev_nav_success = {
            "hold_0.0": False, 
            "hold_1.0": False
        } 
        self._prev_nav_to_pos_success = {
            "hold_0.0": False, 
            "hold_1.0": False
        }
        self._prev_art_success = False
        super().__init__(**kwargs)
    

    # def is_true_from_predicates(self, preds: List[Predicate]) -> bool:
    #     def check_statement(p):
    #         if isinstance(p, LogicalExpr):
    #             return p.is_true_from_predicates(preds)
    #         else:
    #             return p in preds

    #     return self._is_true(check_statement)
    
    
        
    def check_precondition(self, action, cur_node):
        
        def check_logical_expr(p, cur_node):
            if isinstance(p, LogicalExpr):
                return p._is_true(lambda x: check_logical_expr(x, cur_node))
            else: 
                if p.name in ["at", "in"]:
                    p_obj = p._arg_values[0].name
                    p_target = p._arg_values[1].name
                    obj_targ_str = p_obj + "_" + p_target
                    if not cur_node.get(obj_targ_str, False):
                        return False
                
                elif p.name in ["closed_cab", "closed_fridge"]:
                    p_obj = p._arg_values[0].name
                    if not cur_node["closed_" + p_obj]:
                        return False
                
                elif p.name in ["opened_cab", "opened_fridge"]:
                    p_obj = p._arg_values[0].name
                    if not cur_node["opened_" + p_obj]:
                        return False
                
                elif p.name == "robot_at":
                    p_obj = p._arg_values[0].name
                    k = p.name + "_" + p_obj
                
                    if not cur_node.get(k, False):
                        return False
                
                elif p.name == "holding":
                    p_obj = p._arg_values[-1].name
                    if not cur_node["hold_" + p_obj]:
                        return False
                
                elif p.name == "not_holding":
                    p_obj = p._arg_values[-1].name
                    if not cur_node["not_hold_" + p_obj]:
                        return False
            
                return True
        
        is_satisfied =  check_logical_expr(action._pre_cond, cur_node)
        return is_satisfied


        
    def check_pddl_success(self, task_goal, cur_node): 
        for k, v in task_goal.items():
            if k not in cur_node or v != cur_node[k]:
                return False
        return True

    
    def dfs(self, 
            cur_node,
            possible_actions, 
            task_goal, 
            visited: Dict[Tuple, bool] = {}, 
            cur_depth: int = 0, 
            max_depth: int = 8, 
            cur_action_names: List[str] = [],
            verbose: bool = False):

        # true_conds = tuple([k  for k, v in cur_node.items() if v])

        if self.check_pddl_success(task_goal, cur_node):
            if verbose:
                print(f"[SUCCESS]: Depth: {cur_depth} Goal Reached")
            return [], True, cur_depth

        if cur_depth == max_depth:
            if verbose:
                print(f"[FAILED]: Depth: {cur_depth} Max Depth Reached")
            return [], False, 0
        
        found_soln = False
        best_depth = max_depth + 1
        
        for next_action in possible_actions:
            
            if next_action.compact_str in cur_action_names:
                continue

            if cur_depth == 0: 
                if verbose:
                    print("-----------------Start-----------------")
            # if (next_action.name == "nav_to_receptacle" and \
            #     next_action._post_cond[0].compact_str == 'robot_at(fridge_push_point,robot_0)')
            
            if (not self.check_precondition(next_action, cur_node)):
                # if next_action.name == "pick":
                    # breakpoint()
                # self.check_precondition(next_action, cur_node)
                if verbose:
                    print(f"[FAILED]: Depth: {cur_depth} Precond not met: ", next_action.name, next_action._post_cond[0].compact_str)
                continue
                
            next_node = self.update_state(cur_node, next_action)
            if next_node == cur_node:
                if verbose: 
                    print(f"[FAILED]: Depth: {cur_depth} No Change in State: ", next_action.name, next_action._post_cond[0].compact_str)
                continue
            
            # if cur_depth == 2 and next_action.name == "nav":
                # breakpoint()
            
            if verbose:
                print(f"[SUCCESS]: Depth: {cur_depth} Action: ", next_action.name, next_action._post_cond[0].compact_str)
            
            
            next_actions = set(possible_actions) - set([next_action])
            next_action_names = cur_action_names + [next_action.compact_str] 
            action_list, found, depth_found = self.dfs(
                next_node,  
                next_actions, 
                task_goal, 
                visited,
                cur_depth+1,
                max_depth, 
                next_action_names, 
                verbose=verbose
            )
            if found and depth_found < best_depth:
                best_action_list = [next_action] + action_list
                best_depth = depth_found
                found_soln = True

        if found_soln:
            return best_action_list, True, best_depth
        
        else: 
            return [], False, 0
    
    # Tr
    def apply_actions(self, cur_node, action_list):
        next_node = {k: v for k, v in cur_node.items()}
        for action in action_list:
            next_node = self.update_state(next_node, action)
        return next_node
    

    def update_state(self, cur_node, action):
        post_conds = action._post_cond
        next_node = {k: v for k, v in cur_node.items()}            
        for predicate in post_conds:
            if predicate.name in ['at', 'in']: 
                predicate_obj = predicate._arg_values[0].name
                predicate_target = predicate._arg_values[1].name
                for k, v in next_node.items():
                    if predicate_obj in k:
                        next_node[k] = False
                next_node[predicate_obj + "_" + predicate_target] = 1
            
            elif predicate.name in ["closed_cab", "closed_fridge", "opened_cab", "opened_fridge"]:
                predicate_obj = predicate._arg_values[0].name
                next_node[predicate_obj] = True

            elif predicate.name == "robot_at":
                predicate_obj = predicate._arg_values[0].name
                for k, v in next_node.items():
                    if "robot_at" in k:
                        next_node[k] = False
                next_node[predicate.name + "_" + predicate_obj] = True
            
            elif predicate.name == "holding":
                predicate_obj = predicate._arg_values[-1].name
                next_node["hold_" + predicate_obj] = True
                next_node["not_hold_" + predicate_obj] = False
            
            elif predicate.name == "not_holding":
                predicate_obj = predicate._arg_values[-1].name
                next_node["not_hold_" + predicate_obj] = True
                next_node["hold_" + predicate_obj] = False
        return next_node


    def extract_state(self, cur_state, enforce_true=False):
        cur_node = {}
        for predicate in cur_state: 
            if enforce_true:
                p_true = True
            else: 
                p_true = predicate.is_true(self._task.pddl_problem._sim_info)
            
            if predicate.name in ['at', 'in']: 
                predicate_obj = predicate._arg_values[0].name
                predicate_target = predicate._arg_values[1].name
                cur_node[predicate_obj + "_" + predicate_target] = p_true
            
            elif predicate.name in ["closed_cab", "closed_fridge"]:
                predicate_obj = predicate._arg_values[0].name
                cur_node["closed_" + predicate_obj] = p_true

            elif predicate.name in ["opened_cab", "opened_fridge"]:
                predicate_obj = predicate._arg_values[0].name
                cur_node["opened_" + predicate_obj] = p_true


            elif predicate.name == "robot_at":
                predicate_obj = predicate._arg_values[0].name
                cur_node[predicate.name + "_" + predicate_obj] = p_true
            
            elif predicate.name == "holding":
                predicate_obj = predicate._arg_values[-1].name
                cur_node["hold_" + predicate_obj] = p_true
            
            elif predicate.name == "not_holding":
                predicate_obj = predicate._arg_values[-1].name
                cur_node["not_hold_" + predicate_obj] = p_true

            else:
                raise ValueError(f"Predicate {predicate.name} not recognized")

        return cur_node
    
    
    def create_pick_node(self, cur_node): 
        next_node = {k: v for k, v in cur_node.items()}
        next_node["robot_at_goal0|0"] = True
        next_node["not_hold_robot_0"] = True
        next_node["hold_robot_0"] = False
        return next_node
    
    def create_place_node(self, cur_node): 
        next_node = {k: v for k, v in cur_node.items()}
        next_node["robot_at_TARGET_goal0|0"] = True
        next_node["not_hold_robot_0"] = False
        next_node["hold_robot_0"] = True
        return next_node
    
    def create_navt_node(self, cur_node):
        next_node = {k: v for k, v in cur_node.items()}
        next_node["robot_at_TARGET_goal0|0"] = False
        next_node["not_hold_robot_0"] = False
        next_node["hold_robot_0"] = True
        return next_node
    
    def choose_skill_to_execute(self):

        allowed_skills = ["nav", "pick", "place"]
        possible_actions = set(self._task.pddl_problem.get_possible_actions(allowed_action_names = allowed_skills))
        cur_state = self._task.pddl_problem.get_possible_predicates()

        cur_node = {}

        cur_node = self.extract_state(cur_state)
        task_goal = self.extract_state(self._task.pddl_problem.goal._sub_exprs, enforce_true=True)

        # test_pick_node = self.create_pick_node(cur_node)
        # test_place_node = self.create_place_node(cur_node)
        # test_nav_target_node = self.create_navt_node(cur_node)


        # node_dict = {"current: ": cur_node, "pick: ": test_pick_node, "place: ": test_place_node, "navt": test_nav_target_node}
        # for node_k, node_v in node_dict.items():
        #     action_list, found_solution, _ = self.dfs(
        #         node_v, 
        #         possible_actions, 
        #         task_goal, 
        #         visited={}, 
        #         cur_depth=0, 
        #         max_depth=4,
        #         verbose=False
        #     )

        #     print(f"Action List: {node_k}", [action.compact_str for action in action_list])

        action_list, found_solution, _ = self.dfs(
                cur_node, 
                possible_actions, 
                task_goal, 
                visited={}, 
                cur_depth=0, 
                max_depth=4,
                verbose=False
            )
        if not found_solution:
            action_list, found_solution, _= self.dfs(
                cur_node, 
                possible_actions, 
                task_goal, 
                visited={}, 
                cur_depth=0, 
                max_depth=8,
                verbose=True
            )
            return action_list
        
        else: 
            return action_list

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoHierSkillSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(1,), 
            low=-1,
            high=20,
            dtype=np.float32,
        )
    
    def extract_cur_skill_from_plan(self, action_list):
        # assert self._cur_episode_plan is not None, "Episode Plan not found"
        cur_action = action_list[0]
        
        # compact_ep_plan = [action.compact_str for action in self._cur_episode_plan]
        
        # Creating an        
        if cur_action.name == "nav": 
            next_action = action_list[1]
            if next_action.name == "pick":
                return np.array([2.0])
            else: 
                return np.array([4.0])
            
        elif cur_action.name == "pick":
            return np.array([2.0])
    
        else: 
            return np.array([4.0])
            
    
    def get_observation(self, observations, episode, *args, **kwargs):

        is_holding = bool(self._sim.grasp_mgr._snapped_obj_id is not None)        
        nav_to_obj_succ_cls = self._task.measurements.measures.get(MonoNavToObjSuccess.cls_uuid, None)
        nav_to_pos_succ_cls = self._task.measurements.measures.get(NavToPosSucc.cls_uuid, None)
        art_success_cls = self._task.measurements.measures[self._config.art_success_id]
        is_art_task_cls = self._task.measurements.measures.get(AllRankIsArticulatedTaskMeasure.cls_uuid, None)
        dist_to_goal_cls = self._task.measurements.measures.get(DistToGoal.cls_uuid, None)

        nav_to_pos_succ = None
        nav_to_obj_succ = None
        art_obj_success = None
        is_art_task = -1.0

        if self._task.task_type != "rearrange_easy":
            return np.array([-1.0])
        
        if nav_to_obj_succ_cls is not None: 
            nav_to_obj_succ = nav_to_obj_succ_cls.get_metric()
        
        # Extract Navigation to Position Success # 
        if nav_to_pos_succ is not None:
            nav_to_pos_succ = nav_to_pos_succ_cls.get_metric()

        if art_success_cls is not None:
            art_obj_success = art_success_cls.get_metric()

        if is_art_task_cls is not None:
            is_art_task = is_art_task_cls.get_metric()

        if dist_to_goal_cls is not None:
            dist_to_goal = dist_to_goal_cls.get_metric()

        if nav_to_obj_succ is None or is_art_task == -1.0:
            # assert len(observations.keys()) == 1, "Only the head depth should be assigned at the beginning of the episode "
            self._prev_nav_success = {k: False for k in self._prev_nav_success.keys()}
            self._prev_nav_to_pos_success = {k: False for k in self._prev_nav_to_pos_success.keys()}
            self._prev_art_success = False
            # if self._config.use_planner: 
                # self._cur_episode_plan = self.choose_skill_to_execute()
            return np.array([-1.0])

        # Only plan for rearrange_easy tasks #
        
        
        if self._config.use_planner:
            step_planner = self.choose_skill_to_execute()
            if len(step_planner) == 0:
                return np.array([-1.0])
            skill_id = self.extract_cur_skill_from_plan(step_planner)
            return skill_id

        self._prev_nav_success["hold_{:.1f}".format(is_holding)] = nav_to_obj_succ or self._prev_nav_success["hold_{:.1f}".format(is_holding)]
        find_obj_success = dist_to_goal < self._config.nav_find_obj_dist_threshold

            
        self._prev_nav_to_pos_success["hold_{:.1f}".format(is_holding)] = find_obj_success or self._prev_nav_to_pos_success["hold_{:.1f}".format(is_holding)]
        
        if self._task.use_marker_name is not None:
            self._prev_art_success = art_obj_success or self._prev_art_success

        assert isinstance(self._prev_art_success, bool), "Articulated Success is not a boolean"
        assert isinstance(self._prev_nav_success["hold_0.0"], (bool, np.bool_)), "Navigation Success is not a boolean"
        assert isinstance(self._prev_nav_success["hold_1.0"], (bool, np.bool_)), "Navigation Success is not a boolean"
        
        # meta_dist_skill_id = assign_meta_skill_id(
        #     allowed_skills = self._config.allowed_skills, 
        #     is_art_task=is_art_task, 
        #     is_holding=is_holding
        # )
        dist_skill_id = assign_distill_skill_id(
            allowed_skills = self._config.allowed_skills, 
            is_art_task=is_art_task, 
            prev_art_success=self._prev_art_success, 
            prev_nav_success=self._prev_nav_success, 
            prev_nav_to_pos_success = self._prev_nav_to_pos_success, 
            is_holding=is_holding
        )
        return dist_skill_id
        # if meta_dist_skill_id is not None and meta_dist_skill_id[0] != -1.0:
            
        # return dist_skill_id
    

        

@registry.register_measure
class MonoDidPickCorrectObject(Measure):
    cls_uuid: str = "pick_correct_obj_measure"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoDidPickCorrectObject.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self._did_pick = False
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, task, **kwargs):
        did_pick_correct = (self._sim.grasp_mgr._snapped_obj_id is not None) and (self._sim.grasp_mgr._snapped_obj_id == self._sim.scene_obj_ids[task.abs_targ_idx])
        self._did_pick = self._did_pick or did_pick_correct
        self._metric = int(self._did_pick)

@registry.register_measure
class MonoDidPickObjectMeasure(Measure):
    cls_uuid: str = "pick_obj_measure"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoDidPickObjectMeasure.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self._did_pick = False
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        self._did_pick = self._did_pick or (self._sim.grasp_mgr._snapped_obj_id is not None)
        self._metric = int(self._did_pick)


@registry.register_measure
class MonoDidPickMarkerMeasure(Measure):
    cls_uuid: str = "pick_marker_measure"
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        super().__init__(**kwargs)
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoDidPickMarkerMeasure.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self._did_pick = False
        self.update_metric(*args, episode=episode, **kwargs)
    
    def update_metric(self, *args, episode, **kwargs):
        self._did_pick = self._did_pick or (self._sim.grasp_mgr._snapped_marker_id is not None)
        self._metric = int(self._did_pick)

            

            

@registry.register_measure
class MonoArtObjectAtDesiredState(Measure):
    cls_uuid: str = "art_obj_at_desired_state"
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoArtObjectAtDesiredState.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(*args, episode=episode, task=task, observations=observations, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        if task.get_use_marker() is None:
            self._metric = True
        else:
            dist = task.success_js_state - task.get_use_marker().get_targ_js()
            if self._config.use_absolute_distance:
                self._metric = abs(dist) < self._config.success_dist_threshold
            else:
                self._metric = dist < self._config.success_dist_threshold

@registry.register_measure
class MonoEndEffectorDistToMarker(UsesArticulatedAgentInterface, Measure):
    """
    Measures the distance between the end-effector and the target marker.
    """
    cls_uuid: str = "ee_dist_to_marker"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoEndEffectorDistToMarker.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args, 
            episode=episode, 
            task=task, 
            observations=observations, 
            **kwargs
        )
    
    def update_metric(self, *args, episode, task, observations, **kwargs):

        marker = task.get_use_marker()
        if marker is None:
            self._metric = -1.0
        else:
            ee_trans = task._sim.get_agent_data(
                self.agent_id
            ).articulated_agent.ee_transform()
            rel_marker_pos = ee_trans.inverted().transform_point(
                marker.get_current_position()
            )

            self._metric = np.linalg.norm(rel_marker_pos)


@registry.register_measure
class MonoEndEffectorToGoalDistance(UsesArticulatedAgentInterface, Measure):
    cls_uuid: str = "ee_to_goal_distance"

    def __init__(self, sim, *args, **kwargs):
        self._sim = sim
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoEndEffectorToGoalDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, observations, **kwargs):
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )

        if len(self._sim.get_targets()[1]) == 0:
            self._metric = {'0': 0.0}
            return
        
        goals = self._sim.get_targets()[1]

        distances = np.linalg.norm(goals - ee_pos, ord=2, axis=-1)

        self._metric = {str(idx): dist for idx, dist in enumerate(distances)}

@registry.register_measure
class MonoObjectToGoalDistance(Measure):
    """
    Euclidean distance from the target object to the goal.
    """

    cls_uuid: str = "object_to_goal_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoObjectToGoalDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        idxs, goal_pos = self._sim.get_targets()
        if len(idxs) == 0:
            self._metric = {'0': 0.0}
            return
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]
        distances = np.linalg.norm(target_pos - goal_pos, ord=2, axis=-1)
        self._metric = {str(idx): dist for idx, dist in enumerate(distances)}

@registry.register_measure
class MonoEndEffectorToObjectDistance(UsesArticulatedAgentInterface, Measure):
    """
    Gets the distance between the end-effector and all current target object COMs.
    """

    cls_uuid: str = "ee_to_object_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoEndEffectorToObjectDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )

        idxs, _ = self._sim.get_targets()
        if len(idxs) == 0:
            self._metric = {'0': 0.0}
            return
        
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]

        distances = np.linalg.norm(target_pos - ee_pos, ord=2, axis=-1)

        self._metric = {str(idx): dist for idx, dist in enumerate(distances)} 

@registry.register_measure
class MonoRearrangePickSuccess(Measure):
    cls_uuid: str = "pick_success"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._prev_ee_pos = None
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoRearrangePickSuccess.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [MonoEndEffectorToObjectDistance.cls_uuid]
        )
        self._prev_ee_pos = observations["ee_pos"]
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()

        # Is the agent holding the object and it's at the start?
        abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]

        # Check that we are holding the right object and the object is actually
        # being held.
        self._metric = (
            abs_targ_obj_idx == self._sim.grasp_mgr.snap_idx
            and not self._sim.grasp_mgr.is_violating_hold_constraint()
            and ee_to_rest_distance < self._config.ee_resting_success_threshold
        )

        self._prev_ee_pos = observations["ee_pos"]
 
@registry.register_measure
class DidOpenArticulatedReceptacle(Measure):
    cls_uuid = "did_open_art_receptacle"
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DidOpenArticulatedReceptacle.cls_uuid

    def reset_metric(self, task, *args, **kwargs):
        self._metric = False
        task.measurements.check_measure_dependencies(
            self.uuid, [MonoArtObjState.cls_uuid]
        )
        self.update_metric(task, *args, **kwargs)
    
    def update_metric(self, task, *args, episode, **kwargs):
        cur_js_state = task.measurements.measures[MonoArtObjState.cls_uuid].get_metric()
        did_open_art_recep =  cur_js_state > self._config.open_js_threshold * task.success_js_state
        self._metric = self._metric or did_open_art_recep




@registry.register_measure
class MonoRearrangePickReward(RearrangeReward):
    cls_uuid: str = "pick_reward"

    def __init__(self, *args, sim, config, task, **kwargs):
        self.cur_dist = -1.0
        self._prev_picked = False
        self._metric = None

        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoRearrangePickReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                MonoEndEffectorToObjectDistance.cls_uuid,
                RobotForce.cls_uuid,
                ForceTerminate.cls_uuid,
            ],
        )
        self.cur_dist = -1.0
        self._prev_picked = self._sim.grasp_mgr.snap_idx is not None

        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )
        ee_to_object_distance = task.measurements.measures[
            MonoEndEffectorToObjectDistance.cls_uuid
        ].get_metric()
        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()

        snapped_id = self._sim.grasp_mgr.snap_idx
        cur_picked = snapped_id is not None

        if cur_picked:
            dist_to_goal = ee_to_rest_distance
        else:
            dist_to_goal = ee_to_object_distance[str(task.targ_idx)]

        abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]

        did_pick = cur_picked and (not self._prev_picked)
        if did_pick:
            if snapped_id == abs_targ_obj_idx:
                self._metric += self._config.pick_reward
                # If we just transitioned to the next stage our current
                # distance is stale.
                self.cur_dist = -1
            else:
                # picked the wrong object
                self._metric -= self._config.wrong_pick_pen
                if self._config.wrong_pick_should_end:
                    rearrange_logger.debug(
                        "Grasped wrong object, ending episode."
                    )
                    self._task.should_end = True
                self._prev_picked = cur_picked
                self.cur_dist = -1
                return

        if self._config.use_diff:
            if self.cur_dist < 0:
                dist_diff = 0.0
            else:
                dist_diff = self.cur_dist - dist_to_goal

            # Filter out the small fluctuations
            dist_diff = round(dist_diff, 3)
            self._metric += self._config.dist_reward * dist_diff
        else:
            self._metric -= self._config.dist_reward * dist_to_goal
        self.cur_dist = dist_to_goal

        if not cur_picked and self._prev_picked:
            # Dropped the object
            self._metric -= self._config.drop_pen
            if self._config.drop_obj_should_end:
                self._task.should_end = True
            self._prev_picked = cur_picked
            self.cur_dist = -1
            return

        self._prev_picked = cur_picked

@registry.register_measure
class WrongPickMeasure(Measure): 
    cls_uuid: str = "wrong_pick_measure"
    def __init__(self, sim, config, task, *args, **kwargs):
        self._sim = sim
        self._task = task
        super().__init__(**kwargs)
    
    def _get_uuid(self, *args, **kwargs):
        return WrongPickMeasure.cls_uuid
    
    def reset_metric(self, *args, **kwargs):
        self._metric = False
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, **kwargs):
        snapped_id = self._sim.grasp_mgr.snap_idx
        abs_targ_obj_idx = self._sim.scene_obj_ids[self._task.abs_targ_idx]
        if snapped_id is not None and snapped_id != abs_targ_obj_idx:
            self._metric = True

# Terminating on the wrong marker being picked # 
@registry.register_measure
class WrongMarkerPickMeasure(Measure): 
    cls_uuid: str = "wrong_pick_marker_measure"
    def __init__(self, sim, config, task, *args, **kwargs):
        self._sim = sim
        self._task = task
        super().__init__(**kwargs)

    def _get_uuid(self, *args, **kwargs):
        return WrongMarkerPickMeasure.cls_uuid
    
    def reset_metric(self, *args, **kwargs): 
        self._metric = False
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, **kwargs):
        snapped_id = self._sim.grasp_mgr.snapped_marker_id 
        if snapped_id is not None and snapped_id != self._task.use_marker_name:
            self._metric = True

# Difference between the current and previous joint velocity #
@registry.register_measure
class DiffJointVelocity(Measure):
    cls_uuid = "diff_joint_velocity"

    def __init__(self, sim, config, *args, **kwargs):   
        self._sim = sim
        self._config = config
        self._prev_joint_vel = 0.0
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DiffJointVelocity.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._prev_joint_vel = 0.0
        self._metric = 0.0
        self.update_metric(*args, episode=episode, task=task, observations=observations, **kwargs)
    
    # Update this metric during training # 
    def update_metric(self, *args, episode, task, observations, **kwargs):
        if task.use_marker_name is None:
            self._metric = 0.0
            return
        cur_joint_vel = task._sim._markers[task.use_marker_name].get_targ_js_vel()
        add_vel = max(cur_joint_vel - self._prev_joint_vel - self._config.min_vel, 0.0)
        self._metric = add_vel 
        self._prev_joint_vel = cur_joint_vel


@registry.register_measure
class MonoMoveObjectsReward(MoveObjectsReward):
    cls_uuid: str = "move_obj_reward"
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoMoveObjectsReward.cls_uuid

    def update_target_object(self):
        """
        The agent just finished one rearrangement stage so it's time to
        update the target object for the next stage.
        """
        # Get the next target object
        idxs, _ = self._sim.get_targets()
        if len(idxs) != 0:
            targ_obj_idx = idxs[self._cur_rearrange_stage]
        else: 
            targ_obj_idx = 0
        # Get the target object's absolute index
        self.abs_targ_obj_idx = self._sim.scene_obj_ids[targ_obj_idx]

    # Resetting task metric # 
    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._prev_holding_marker = False
        self._gave_marker_reward = {}
        self._prev_arm_action = None
        self._prev_base_action = None
        self._prev_joint_vel = 0.0
        
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                DiffJointVelocity.cls_uuid,
            ],
        )
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )
        
        
    # Update metric during training # 
    def update_metric(self, *args, episode, task, observations, **kwargs):
        
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

        joint_vel_reward = task.measurements.measures[
            DiffJointVelocity.cls_uuid
        ].get_metric()

        cur_arm_action = task.actions["arm_action"].cur_action
        if self._prev_arm_action is not None: 
            self._metric += (
                -1.0 * self._config.action_pen * np.linalg.norm(cur_arm_action - self._prev_arm_action) 
            )
        
        self._prev_arm_action = cur_arm_action
        
        # Use marker name is None, then we don't need to check for marker reward
        if task.use_marker_name is not None:
            is_holding_marker = self._sim.grasp_mgr.snapped_marker_id == task.use_marker_name 
            picked_up_marker = is_holding_marker and not self._prev_holding_marker
            already_gave_reward = (
                self._cur_rearrange_stage in self._gave_marker_reward
            )
            if picked_up_marker and not already_gave_reward:
                self._metric += self._config.pick_marker_reward
                self._gave_marker_reward[self._cur_rearrange_stage] = True

            targ_js_vel = task._sim._markers[task.use_marker_name].get_targ_js_vel()    
            self._metric += -1.0 * self._config.js_vel_pen * joint_vel_reward
            self._prev_joint_vel = targ_js_vel
        
       

        
@registry.register_measure
class MonoPddlStageGoals(Measure):
    _stage_succ: List[str]
    cls_uuid: str = "mono_pddl_stage_goals"
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoPddlStageGoals.cls_uuid
    def reset_metric(self, *args, **kwargs):
        self._stage_succ = []
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = {}
        if hasattr(task, 'task_type') and task.task_type == "nav_pick_nav_place":
            for stage_name, logical_expr in task.pddl_problem.stage_goals.items():
                succ_k = f"{stage_name}_success"
                if stage_name in self._stage_succ:
                    self._metric[succ_k] = 1.0
                else:
                    if task.pddl_problem.is_expr_true(logical_expr):
                        self._metric[succ_k] = 1.0
                        self._stage_succ.append(stage_name)
                    else:
                        self._metric[succ_k] = 0.0


@registry.register_measure
class MonoArtObjReward(RearrangeReward):
    """
    A general reward definition for any tasks involving manipulating articulated objects.
    """

    cls_uuid: str = "art_obj_reward"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._metric = None

        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoArtObjReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                MonoArtObjState.cls_uuid,
                MonoArtObjSuccess.cls_uuid,
                EndEffectorToRestDistance.cls_uuid,
                MonoArtObjectAtDesiredState.cls_uuid,
            ],
        )
        link_state = task.measurements.measures[
            MonoArtObjState.cls_uuid
        ].get_metric()

        dist_to_marker = task.measurements.measures[
            MonoEndEffectorDistToMarker.cls_uuid
        ].get_metric()

        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()

        self._prev_art_state = link_state
        self._any_has_grasped = task._sim.grasp_mgr.is_grasped
        self._prev_ee_dist_to_marker = dist_to_marker
        self._prev_ee_to_rest = ee_to_rest_distance
        self._any_at_desired_state = False
        self._prev_joint_vel = 0.0
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )
        reward = self._metric
        link_state = task.measurements.measures[
            ArtObjState.cls_uuid
        ].get_metric()

        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()

        is_art_obj_state_succ = task.measurements.measures[
            MonoArtObjectAtDesiredState.cls_uuid
        ].get_metric()

        cur_dist = abs(link_state - task.success_js_state)
        prev_dist = abs(self._prev_art_state - task.success_js_state)

        dist_diff = prev_dist - cur_dist
        if not is_art_obj_state_succ:
            reward += self._config.art_dist_reward * dist_diff

        cur_has_grasped = task._sim.grasp_mgr.is_grasped

        cur_ee_dist_to_marker = task.measurements.measures[
            EndEffectorDistToMarker.cls_uuid
        ].get_metric()

        diff_joint_velocity = task.measurements.measures[
            DiffJointVelocity.cls_uuid
        ].get_metric()

        # If the task is an articulated one, use the marker # 
        if task.use_marker_name is not None:
            if cur_has_grasped and not self._any_has_grasped:
                if task._sim.grasp_mgr.snapped_marker_id != task.use_marker_name:
                    # Grasped wrong marker
                    reward -= self._config.wrong_grasp_pen
                    if self._config.wrong_grasp_end:
                        rearrange_logger.debug(
                            "Grasped wrong marker, ending episode."
                        )
                        task.should_end = True
                else:
                    # Grasped right marker
                    reward += self._config.grasp_reward
                self._any_has_grasped = True
            
            targ_js_vel = task._sim._markers[task.use_marker_name].get_targ_js_vel()    
            self._metric += -1.0 * self._config.js_vel_pen * diff_joint_velocity
            self._prev_joint_vel = targ_js_vel

        if is_art_obj_state_succ:
            if not self._any_at_desired_state:
                reward += self._config.art_at_desired_state_reward
                self._any_at_desired_state = True
            # Give the reward based on distance to the resting position.
            ee_dist_change = self._prev_ee_to_rest - ee_to_rest_distance
            reward += self._config.ee_dist_reward * ee_dist_change
        elif not cur_has_grasped:
            # Give the reward based on distance to the handle
            dist_diff = self._prev_ee_dist_to_marker - cur_ee_dist_to_marker
            reward += self._config.marker_dist_reward * dist_diff

        self._prev_ee_to_rest = ee_to_rest_distance

        self._prev_ee_dist_to_marker = cur_ee_dist_to_marker
        self._prev_art_state = link_state
        self._metric = reward

# Mono Articulated Object State # 
@registry.register_measure
class MonoArtObjState(Measure):
    """
    Measures the current joint state of the target articulated object.
    """
    cls_uuid: str = "art_obj_state"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoArtObjState.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        if task.get_use_marker() is None:
            self._metric = -1.0
        else:
            self._metric = task.get_use_marker().get_targ_js()

@registry.register_measure
class MonoArtObjSuccess(Measure):
    """
    Measures if the target articulated object joint state is at the success criteria.
    """

    cls_uuid: str = "art_obj_success"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        self._sim = sim
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoArtObjSuccess.cls_uuid

    # Reset Metric #
    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    # Update Metric #
    def update_metric(self, *args, episode, task, observations, **kwargs):
        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()
        is_art_obj_state_succ = task.measurements.measures[
            MonoArtObjectAtDesiredState.cls_uuid
        ].get_metric()

        self._metric = (
            is_art_obj_state_succ
            and ee_to_rest_distance < self._config.rest_dist_threshold
            and not self._sim.grasp_mgr.is_grasped
        )
      

@registry.register_measure
class CompositePddlSuccess(Measure):
    cls_uuid: str = "composite_pddl_success"
    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(**kwargs)
        self._sim = sim
        
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return CompositePddlSuccess.cls_uuid
    
    def reset_metric(self, *args, task,  **kwargs):
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs): 
        # if task.measurements.measures['obj_at_goal.0'] == 1:
            # breakpoint()
        if hasattr(task, 'task_type') and task.task_type == "rearrange_easy":
            self._metric = task.pddl_problem.is_expr_true(task.pddl_problem.goal)
        else: 
            self._metric = False



@registry.register_measure
class ObjAtGoalPddlSuccess(Measure):
    cls_uuid: str = "obj_at_goal_pddl_success"
    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(**kwargs)
        self._sim = sim

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjAtGoalPddlSuccess.cls_uuid
    
    def reset_metric(self, *args, task,  **kwargs):
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        if hasattr(task, 'task_type') and task.task_type == "rearrange_easy":
            holding_obj = self._sim.grasp_mgr.is_grasped
            self._metric = task.measurements.measures['obj_at_goal'].get_metric()['0'] and not holding_obj
        else:
            self._metric = False

# @registry.register_measure
# class MonoRotDistToGoal(RotDistToGoal):
#     def update_metric(self, *args, episode, task, observations, **kwargs):
#         super().update_metric(*args, episode, task, observations, **kwargs)

# @registry.register_measure
# class MonoDistToGoal(DistToGoal): 
#     def update_metric(self, *args, episode, task, observations, **kwargs):
#         super().update_metric(*args, episode, task, observations, **kwargs)

@registry.register_sensor
class MonoMarkerRelPosSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Tracks the relative position of a marker to the robot end-effector
    specified by `use_marker_name` in the task. This `use_marker_name` must
    exist in the task and refer to the name of a marker in the simulator.
    """

    cls_uuid: str = "marker_rel_pos"

    def __init__(self, sim, config, *args, task, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._task = task

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MonoMarkerRelPosSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        marker = self._task.get_use_marker()
        if marker is not None:
            ee_trans = self._sim.get_agent_data(
                self.agent_id
            ).articulated_agent.ee_transform()
            rel_marker_pos = ee_trans.inverted().transform_point(
                marker.get_current_position()
            )
        else: 
            rel_marker_pos = np.zeros(3)

        return np.array(rel_marker_pos)
