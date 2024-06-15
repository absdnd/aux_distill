
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    NavToObjSuccess,
    RotDistToGoal, 
    DistToGoal, 
    DoesWantTerminate, 
    NavToPosSucc,
    NavToObjReward,
)
from habitat.tasks.rearrange.rearrange_sensors import (
    DoesWantTerminate,
    RearrangeReward,
)
from habitat.core.embodied_task import Measure

# What is my rotation distance to goal?
@registry.register_measure
class AuxRotDistToGoal(RotDistToGoal):
    cls_uuid: "aux_rot_dist_to_goal"
    def __init__(self, *args, config, **kwargs):
        self._config = config
        super().__init__(*args, config=config, **kwargs)
   
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return AuxRotDistToGoal.cls_uuid
    
    def update_metric(self, *args, episode, task, observations, **kwargs):
        if task._config.type in self._config.allowed_tasks:
            super().update_metric(
                *args, 
                episode = episode, 
                task = task, 
                observations = observations, 
                **kwargs
            )
        else: 
            self._metric = 0.0

@registry.register_measure
class AuxDistToGoal(DistToGoal): 
    cls_uuid: str = "aux_dist_to_goal"
    def __init__(self, *args, config, **kwargs):
        self._config = config
        super().__init__(*args, config=config, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return AuxDistToGoal.cls_uuid
    
    def update_metric(self, *args, episode, task, observations, **kwargs):
        if task._config.type in self._config.allowed_tasks:
            super().update_metric(
                *args, 
                episode = episode, 
                task = task, 
                observations = observations, 
                **kwargs
            )
        else: 
            self._metric = 0.0


@registry.register_measure
class AuxNavToPosSucc(NavToPosSucc):
    uuid: str = "aux_nav_to_pos_succ"
    def update_metric(self, *args, episode, task, observations, **kwargs):
        dist = task.measurements.measures[AuxDistToGoal.cls_uuid].get_metric()
        self._metric = dist < self._config.success_distance
    
    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [AuxDistToGoal.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return AuxNavToPosSucc.uuid

@registry.register_measure
class DidAuxNavToPosSuccess(Measure):
    cls_uuid: str = "did_nav_to_pos_success"
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._num_steps = {}
        super().__init__(**kwargs)
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DidAuxNavToPosSuccess.cls_uuid
    
    def reset_metric(self, task, *args, **kwargs):
        self._metric = {
            "goal": False,
            "target": False,
        }
        self._num_steps = {
            "goal": 0,
            "target": 0,
        }
        task.measurements.check_measure_dependencies(
            self.uuid,
            [AuxNavToPosSucc.cls_uuid],
        )
        self.update_metric(task, *args, **kwargs)
    
    def update_metric(self, task, observations, *args, **kwargs):
        nav_to_obj_succ = task.measurements.measures[AuxNavToPosSucc.cls_uuid].get_metric()
        num_steps=  task.measurements.measures["num_steps"].get_metric()
        is_holding = self._sim.grasp_mgr.snap_idx is not None
        if nav_to_obj_succ:
            if is_holding==0:
                self._metric["goal"] = self._metric["goal"] or nav_to_obj_succ
                self._num_steps["goal"] = num_steps
            else:
                # breakpoint()
                self._num_steps["target"] = num_steps
                self._metric["target"] = self._metric["target"] or nav_to_obj_succ

# Did Navigate to Objec Success
@registry.register_measure
class DidAuxNavToObjSuccess(Measure):
    cls_uuid: str = "did_nav_to_obj_success"
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DidAuxNavToObjSuccess.cls_uuid
    
    def reset_metric(self, task, *args, **kwargs):
        self._metric = {
            "goal": False,
            "target": False,
        }
        task.measurements.check_measure_dependencies(
            self.uuid,
            [AuxNavToObjSuccess.cls_uuid],
        )
        self.update_metric(task, *args, **kwargs)
    
    def update_metric(self, task, observations, *args, **kwargs):
        nav_to_obj_succ = task.measurements.measures[AuxNavToObjSuccess.cls_uuid].get_metric()
        is_holding = observations["is_holding"][0]
        if nav_to_obj_succ:
            if is_holding==0:
                self._metric["goal"] = self._metric["goal"] or nav_to_obj_succ
            else:
                self._metric["target"] = self._metric["target"] or nav_to_obj_succ



@registry.register_measure
class AuxNavToObjReward(RearrangeReward):
    cls_uuid: str = "Aux_nav_to_obj_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return AuxNavToObjReward.cls_uuid


    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                AuxNavToObjSuccess.cls_uuid,
                DistToGoal.cls_uuid,
                RotDistToGoal.cls_uuid,
            ],
        )
        self._cur_angle_dist = -1.0
        self._prev_dist = -1.0
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        reward = 0.0
        cur_dist = task.measurements.measures[DistToGoal.cls_uuid].get_metric()
        if self._prev_dist < 0.0:
            dist_diff = 0.0
        else:
            dist_diff = self._prev_dist - cur_dist

        reward += self._config.dist_reward * dist_diff
        self._prev_dist = cur_dist

        if (
            self._config.should_reward_turn
            and cur_dist < self._config.turn_reward_dist
        ):
            angle_dist = task.measurements.measures[
                AuxRotDistToGoal.cls_uuid
            ].get_metric()

            if self._cur_angle_dist < 0:
                angle_diff = 0.0
            else:
                angle_diff = self._cur_angle_dist - angle_dist

            reward += self._config.angle_dist_reward * angle_diff
            self._cur_angle_dist = angle_dist

        self._metric = reward

@registry.register_measure  
class AuxNavToObjSuccess(NavToObjSuccess):
    cls_uuid: str = "Aux_nav_to_obj_success"
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return AuxNavToObjSuccess.cls_uuid
    
    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [NavToPosSucc.cls_uuid, RotDistToGoal.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        angle_dist = task.measurements.measures[
            RotDistToGoal.cls_uuid
        ].get_metric()

        nav_pos_succ = task.measurements.measures[
            NavToPosSucc.cls_uuid
        ].get_metric()

        
        # Must look at target object #
        if self._config.must_look_at_targ:
            self._metric = (
                nav_pos_succ and angle_dist < self._config.success_angle_dist
            )
        else:
            self._metric = nav_pos_succ

        if self._config.must_call_stop:
            called_stop = task.measurements.measures[DoesWantTerminate.cls_uuid].get_metric()
            if called_stop:
                task.should_end = True
            else:
                self._metric = False