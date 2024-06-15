from habitat.tasks.rearrange.actions.actions import ArmAction, BaseVelAction
from habitat.core.registry import registry
import numpy as np

# Monolithic arm action # 
@registry.register_task_action
class MonoArmAction(ArmAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cur_action = None 
    
    @property
    def cur_action(self):
        return self._cur_action
    
    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        self._cur_action = kwargs[self._action_arg_prefix + "arm_action"]

@registry.register_task_action
class MonoBaseVelocityAction(BaseVelAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cur_action = None 
    
    @property
    def cur_action(self):
        return self._cur_action
    
    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        self._cur_action = np.array(
            kwargs[self._action_arg_prefix + "base_vel"]
        )