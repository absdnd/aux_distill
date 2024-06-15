
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.pick_task import RearrangePickTaskV1
from typing import Optional
import numpy as np

@registry.register_task(name="MonoRearrangePickTask-v0")
class MonoRearrangePickTaskV1(RearrangePickTaskV1):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            **kwargs,
        )
        self.task_type = "mono_pick"
        self.use_marker_name = None
        self.task_id = self._config.task_id
    
    def get_use_marker(self):
        return None
    
    def get_sampled(self):
        return [None]
    
    @property
    def success_js_state(self) -> float:
        """
        The success state of the articulated object desired joint.
        """
        return self._config.success_state
    
    @property
    def nav_goal_pos(self):
        return np.array([0.,0.,0.])


# The cabinet is closed and the agent has to pick up the object # 
@registry.register_task(name="MonoClosedCabinetRearrangePickTask-v0")
class MonoClosedCabinetPickTask(MonoRearrangePickTaskV1):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            **kwargs,
        )
        self.task_type = "fridge_pick"
        self._use_marker = None
        self.task_id = self._config.task_id
    
    def get_use_marker(self):
        """
        The marker the agent should interact with.
        """
        return self._sim.get_marker(self._use_marker)
    
    def _disable_art_sleep(self):
        """
        Disables the sleeping state of the articulated object. Use when setting
        the articulated object joint states.
        """
        ao = self.get_use_marker().ao_parent
        self._prev_awake = ao.awake
        ao.awake = True
    
    def _reset_art_sleep(self) -> None:
        """
        Resets the sleeping state of the target articulated object.
        """
        ao = self.get_use_marker().ao_parent
        ao.awake = self._prev_awake
    
    @property
    def success_js_state(self) -> float:
        """
        The success state of the articulated object desired joint.
        """
        return self._config.success_state
    
    @property
    def nav_goal_pos(self):
        return np.array([0.,0.,0.])

    def _gen_start_state(self):
        return np.array([0, 3 * np.pi/2])

    # Set link state of articulated agent # 
    def _set_link_state(self, art_pos: np.ndarray) -> None:
        """
        Set the joint state of all the joints on the target articulated object.
        """
        ao = self.get_use_marker().ao_parent
        ao.joint_positions = art_pos

    
    # Is the episode valid or not? # 
    def _check_episode_is_valid(self, episode: Episode):
        ep_target = [*episode.targets][0]
        receptacle_name =episode.name_to_receptacle[ep_target]
        if "refrigerator" not in receptacle_name:
            raise RuntimeError("Episode is not valid for this task")
        
    # Set all receptacles to be open
    def reset(self, episode: Episode, *args, **kwargs):
        # self._check_episode_is_valid(episode)
        self._disable_art_sleep()
        self._set_link_state(self._gen_start_state())   
        self._sim.internal_step(-1)
        self._reset_art_sleep()    
        return super().reset(episode, *args, **kwargs)
           


# Learning to pick from an open fridge # 
@registry.register_task(name="MonoFridgeRearrangePickTask-v0")
class MonoFridgeRearrangePickTaskV1(MonoRearrangePickTaskV1):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            **kwargs,
        )
        self.task_type = "fridge_pick"
        self._use_marker = "fridge_push_point"
        self.task_id = self._config.task_id
    
    def get_use_marker(self):
        """
        The marker the agent should interact with.
        """
        return self._sim.get_marker(self._use_marker)
    
    def _disable_art_sleep(self):
        """
        Disables the sleeping state of the articulated object. Use when setting
        the articulated object joint states.
        """
        ao = self.get_use_marker().ao_parent
        self._prev_awake = ao.awake
        ao.awake = True
    
    def _reset_art_sleep(self) -> None:
        """
        Resets the sleeping state of the target articulated object.
        """
        ao = self.get_use_marker().ao_parent
        ao.awake = self._prev_awake
    
    @property
    def success_js_state(self) -> float:
        """
        The success state of the articulated object desired joint.
        """
        return self._config.success_state
    
    @property
    def nav_goal_pos(self):
        return np.array([0.,0.,0.])

    def _gen_start_state(self):
        return np.array([0, 3 * np.pi/2])

    # Set link state of articulated agent # 
    def _set_link_state(self, art_pos: np.ndarray) -> None:
        """
        Set the joint state of all the joints on the target articulated object.
        """
        ao = self.get_use_marker().ao_parent
        ao.joint_positions = art_pos

    
    # Is the episode valid or not? # 
    def _check_episode_is_valid(self, episode: Episode):
        ep_target = [*episode.targets][0]
        receptacle_name =episode.name_to_receptacle[ep_target]
        if "refrigerator" not in receptacle_name:
            raise RuntimeError("Episode is not valid for this task")
        
    # Set all receptacles to be open
    def reset(self, episode: Episode, *args, **kwargs):
        # self._check_episode_is_valid(episode)
        self._disable_art_sleep()
        self._set_link_state(self._gen_start_state())   
        self._sim.internal_step(-1)
        self._reset_art_sleep()    
        return super().reset(episode, *args, **kwargs)
           

