import sys
import os
sys.path.append(os.getcwd())
from skill_chain.trainer.custom_gym_env import CustomVectorEnvFactory
from skill_chain.trainer.aux_trainer import AuxDistillPPOTrainer

from skill_chain.policy.aux_mtask_policy import AuxMultiTaskPolicy
# from skill_chain.tasks.mono_place_task import NoDropPlaceTaskV1
from skill_chain.tasks.custom_pddl_task import CustomPddlTask
from skill_chain.tasks.mono_pick_task import MonoRearrangePickTaskV1
from skill_chain.tasks.mono_place_task import MonoRearrangePlaceTaskV1

import skill_chain.sensors.mono_nav_to_obj_sensors
import skill_chain.sensors.mono_sensors
import skill_chain.trainer.aux_ppo
from skill_chain.trainer.mono_access_mgr import MonoAgentAccessMgr
from skill_chain.tasks.mono_nav_task import MonoNavToObjTask
from skill_chain.tasks.mono_articulated_object_task import MonoRearrangeOpenFridgeTaskV1
# from skill_chain.simulator.dataset import RearrangeHardDatasetV1
from skill_chain.actions.mono_actions import MonoArmAction, MonoBaseVelocityAction
from skill_chain.tasks.mono_lang_pick_task import PddlMultiTask