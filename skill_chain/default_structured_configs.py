"""
Contains the structured config definitions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from habitat.config.default_structured_configs import (
    MeasurementConfig, 
    TaskConfig, 
    NavToPosSuccMeasurementConfig,
    NavToObjRewardMeasurementConfig,
    NavToObjSuccessMeasurementConfig,
    ArtObjSuccessMeasurementConfig,
    ArtObjRewardMeasurementConfig,
    ArtObjStateMeasurementConfig,
    ArtObjAtDesiredStateMeasurementConfig,
    RearrangePickRewardMeasurementConfig, 
    IteratorOptionsConfig,
    EnvironmentConfig,
    HabitatConfig,
    MoveObjectsRewardMeasurementConfig,
    ArmActionConfig,
    BaseVelocityActionConfig, 
    MarkerRelPosSensorConfig,
)
from habitat.config.default_structured_configs import LabSensorConfig
from habitat_baselines.config.default_structured_configs import WBConfig, RLConfig, HabitatBaselinesRLConfig, PPOConfig, HabitatBaselinesConfig
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from skill_configs import DistillTaskConfig

cs = ConfigStore.instance()



@dataclass
class CustomWeightsandBiasesConfig(WBConfig):
    sweep_name: str =  ""
    sweep_param: str = ""
    sweep_type: str = ""

@dataclass
class CustomMonoMeasureConfig(MeasurementConfig):
    type: str = "EnvTaskMeasure"
    allowed_skills: List = field(
        default_factory=list
    )

@dataclass
class IsArtMeasureConfig(MeasurementConfig):
    type: str = "IsArticulatedTaskMeasure"

# @dataclass
# c
@dataclass
class CustomMonoSensorConfig(LabSensorConfig):
    type: str = "EnvTaskSensor"

@dataclass
class IsArtSensorConfig(LabSensorConfig):
    type: str = "IsArticulatedSensor"

@dataclass
class AllRankIsArtMeasureConfig(MeasurementConfig):
    type: str = "AllRankIsArticulatedTaskMeasure"
@dataclass
class CustomSeqTaskConfig():
    use_seq_skills: bool =  False
    before_grasp_id: int = 2
    after_grasp_id: int = 4

@dataclass
class MonoNavToObjSuccessMeasurementConfig(NavToObjSuccessMeasurementConfig):
    type: str = "MonoNavToObjSuccess"
@dataclass
class DidMonoNavToObjSuccessMeasurementConfig(MeasurementConfig):
    type: str = "DidMonoNavToObjSuccess"

@dataclass
class DidMonoNavToPosSuccessMeasurementConfig(MeasurementConfig):
    type: str = "DidMonoNavToPosSuccess"

@dataclass
class DidOpenArticulatedReceptacle(MeasurementConfig):
    type: str = "DidOpenArticulatedReceptacle"
    open_js_threshold: float = 0.25

@dataclass
class MonoMarkerRelPosSensorConfig(MarkerRelPosSensorConfig):
    type: str = "MonoMarkerRelPosSensor"

@dataclass
class MonoGoalSensorConfig(LabSensorConfig):
    """
    Rearrangement only. Returns the relative position from end effector to a goal position in which the agent needs to place an object.
    """

    type: str = "MonoGoalSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@dataclass
class MonoArtObjSuccessMeasurementConfig(ArtObjSuccessMeasurementConfig):
    type: str = "MonoArtObjSuccess"

@dataclass
class MonoArtSuccessPosMeasurementConfig(MeasurementConfig):
    type: str = "MonoArtSuccessPos"
@dataclass
class MonoArtObjStateMeasurementConfig(ArtObjStateMeasurementConfig): 
    type: str = "MonoArtObjState"

@dataclass
class MonoEndEffectorDistToMarkerConfig(MeasurementConfig):
    type: str = "MonoEndEffectorDistToMarker"

@dataclass
class MonoArtObjAtDesiredStateMeasurementConfig(ArtObjAtDesiredStateMeasurementConfig): 
    type: str = "MonoArtObjectAtDesiredState"

@dataclass
class MonoArtObjRewardMeasurementConfig(ArtObjRewardMeasurementConfig):
    type: str = "MonoArtObjReward"
    js_vel_pen: float = 0.0

@dataclass
class DiffJointVelocityMeasurementConfig(MeasurementConfig):
    type: str = "DiffJointVelocity"
    min_vel: float = 0.05

@dataclass 
class MonoDidPickObjectMeasurementConfig(MeasurementConfig):
    type: str = "MonoDidPickObjectMeasure"

@dataclass
class MonoDidPickCorrectObjectMeasurementConfig(MeasurementConfig):
    type: str = "MonoDidPickCorrectObject"
@dataclass
class MonoDidPickMarkerMeasurementConfig(MeasurementConfig):
    type: str = "MonoDidPickMarkerMeasure"

@dataclass
class MonoArmActionConfig(ArmActionConfig):
    type: str = "MonoArmAction"

@dataclass
class MonoBaseVelocityActionConfig(BaseVelocityActionConfig):
    type: str = "MonoBaseVelocityAction"

@dataclass
class CustomTaskConfig(TaskConfig):
    start_template: Optional[List[str]] = MISSING
    goal_template: Optional[Dict[str, Any]] = MISSING
    sample_entities: Dict[str, Any] = MISSING
    target_sampling_strategy: str = (
        "object_type"  # object_instance or object_type
    )
    target_type: str = "object_type"  # object_instance or object_type
    seq_task_config: CustomSeqTaskConfig = CustomSeqTaskConfig()
    task_id: int = -1
    seq_skills: bool =  False
    should_prevent_drop: bool = False
    should_prevent_stop: bool = False
    stop_at_goal: bool = False
    no_arm_action: bool = False
    spawn_agent_at_min_distance: bool = False
    assign_marker_to_open_recep: bool = False
    open_ratio: float = 1.0
    use_marker_options: List[str] = field(default_factory=lambda: [])

@dataclass
class MonoHierSkillMeasurementConfig(MeasurementConfig):
    type: str = "MonoHierSkillMeasure"
    

@dataclass
class EnvTaskSensorConfig(LabSensorConfig):
    type: str = "EnvTaskSensor"
    max_skills: int = 6 # Nav, Pick, Place, Nav + Pick, Nav + place, Nav + Pick + Nav + Place


@dataclass
class MonoRearrangePickRewardMeasurementConfig(RearrangePickRewardMeasurementConfig):
    r"""
    Rearrangement Only. Requires the end_effector_sensor lab sensor. The reward for the pick task.

    :property dist_reward: At each step, the measure adds dist_reward times the distance the end effector moved towards the target.
    :property pick_reward: If the robot picks the target object, it receives pick_reward reward.
    :property drop_pen: The penalty for dropping the object.
    :property wrong_pick_pen: The penalty for picking the wrong object.
    :property force_pen: At each step, adds a penalty of force_pen times the current force on the robot.
    :property drop_obj_should_end: If true, the task will end if the robot drops the object.
    :property wrong_pick_should_end: If true, the task will end if the robot picks the wrong object.
    """
    type: str = "MonoRearrangePickReward"


@dataclass
class MonoTargetStartSensorConfig(LabSensorConfig):
    r"""
    Rearrangement only. Returns the relative position from end effector to a target object that needs to be picked up.
    """
    type: str = "MonoTargetStartSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3
    
@dataclass
class CustomPddlSuccessConfig(MeasurementConfig):
    type: str = "CompositePddlSuccess"

@dataclass
class ObjAtGoalPddlSuccessConfig(MeasurementConfig):
    type: str = "ObjAtGoalPddlSuccess"

@dataclass
class DistLossConfig:
    type: str = "dist_loss"
    dist_loss_coef: float = 0.0
    dist_skill_id: int = 5

@dataclass
class CustomPPOConfig(PPOConfig):
    dist_loss_coef: float = 0.0
    dist_skill_id: int = 5

class CustomRLConfig(RLConfig):
    ppo: CustomPPOConfig = CustomPPOConfig()

@dataclass
class OneHotTargetSensorConfig(LabSensorConfig):
    type: str = "OneHotTargetSensor"


@dataclass
class CustomHabitatBaselinesRLConfig(HabitatBaselinesConfig):
    rl: CustomRLConfig = CustomRLConfig()



@dataclass
class MonoPddlStageGoalsConfig(MeasurementConfig):
    type: str = "MonoPddlStageGoals"

@dataclass
class WrongPickMeasureConfig(MeasurementConfig):
    type: str = "WrongPickMeasure"

@dataclass
class WrongMarkerPickMeasureConfig(MeasurementConfig):
    type: str = "WrongMarkerPickMeasure"

@dataclass
class MonoRotDistToGoalConfig(MeasurementConfig):
    type: str = "MonoRotDistToGoal"
    allowed_tasks: List[str] = ("NavToObjTask-v0", "CustomRearrangePddlTask-v0", "RearrangePddlTask-v0")

@dataclass
class MonoDistToGoalConfig(MeasurementConfig):
    type: str = "MonoDistToGoal"
    allowed_tasks: List[str] = ("NavToObjTask-v0", "CustomRearrangePddlTask-v0", "RearrangePddlTask-v0")

@dataclass
class MonoNavToPosSuccConfig(NavToPosSuccMeasurementConfig):
    type: str = "MonoNavToPosSucc"

@dataclass
class MonoNavToObjRewardConfig(NavToObjRewardMeasurementConfig):
    type: str = "MonoNavToObjReward"

@dataclass
class MonoRearrangePickSuccessMeasurementConfig(MeasurementConfig):
    r"""
    Rearrangement Only. Requires the end_effector_sensor lab sensor. 1.0 if the robot picked the target object.
    """
    type: str = "MonoRearrangePickSuccess"
    ee_resting_success_threshold: float = 0.15

@dataclass
class MonoIteratorOptionsConfig(IteratorOptionsConfig):
    filter_criteria: List[str] = field(default_factory=lambda: ["art", "no_art"])

@dataclass
class MonoEndEffectorToObjectDistanceMeasurementConfig(MeasurementConfig):
    type: str = "MonoEndEffectorToObjectDistance"

@dataclass
class MonoEndEffectorToGoalDistanceMeasurementConfig(MeasurementConfig):
    type: str = "MonoEndEffectorToGoalDistance"

@dataclass
class MonoObjectToGoalDistanceMeasurementConfig(MeasurementConfig):
    type: str = "MonoObjectToGoalDistance"

# @dataclass
# class MonoAllowedSkillsMeasurementConfig(MeasurementConfig):
#     type: str = "MonoAllowedSkills"
#     allowed_skills: List[str] = field(default_factory=lambda: [])


# Adding in Mono-Environment Configuration # 
@dataclass 
class MonoEnvironmentConfig(EnvironmentConfig):
    max_episode_steps: int = 1000
    max_episode_seconds: int = 10000000
    # iterator_options: MonoIteratorOptionsConfig = MonoIteratorOptionsConfig()
    iterator_options: IteratorOptionsConfig = IteratorOptionsConfig()

@dataclass 
class MonoHabitatConfig(HabitatConfig):
    environment: MonoEnvironmentConfig = MonoEnvironmentConfig()

@dataclass
class MonoMoveObjectsRewardMeasurementConfig(MoveObjectsRewardMeasurementConfig):
    type: str = "MonoMoveObjectsReward"
    pick_marker_reward: float = 0.0
    action_pen: float = 0.0
    js_vel_pen: float = 0.0

@dataclass
class MonoTaskStageStepsMeasurementConfig(MeasurementConfig):
    type: str = "MonoTaskStageSteps"
    allowed_skills: List[str] = field(default_factory=lambda: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    allowed_skill_ids: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    max_episode_steps: List[int] = field(
        default_factory=lambda: [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
    )
    force_early_end: bool = True





@dataclass
class MonoHierSkillSensorConfig(LabSensorConfig):
    type: str = "MonoHierSkillSensor"
    art_success_id: str = "art_obj_at_desired_state" 
    art_success_dist_threshold: float = 0.30
    allowed_skills: List[str] = field(default_factory=lambda:["nav_pick", "nav_place", "nav_open_cabinet", "nav_open_fridge", "nav_pick_nav_place"])
    nav_find_obj_dist_threshold: float = 3.0
    use_planner: bool = False

@dataclass
class SkillConfig():
    pick: DistillTaskConfig = MISSING
    nav_pick: DistillTaskConfig = MISSING
    nav_place: DistillTaskConfig = MISSING
    nav_pick_nav_place: DistillTaskConfig = MISSING
    nav_open_fridge: DistillTaskConfig = MISSING
    nav_open_cabinet: DistillTaskConfig = MISSING
    open_fridge_pick: DistillTaskConfig = MISSING
    lang_pick_prox: DistillTaskConfig = MISSING

@dataclass
class EnvTasksConfig():
    random_seed_by_rank: bool =True
    eval_allowed_skills:List[str]=field(
        default_factory=lambda:["nav_pick","nav_place","nav_pick_nav_place","nav_open_fridge","nav_open_cabinet","open_fridge_pick"]
    )
    allowed_skills: List[str] = field(
        default_factory=lambda:["nav_pick","nav_place","nav_pick_nav_place","nav_open_fridge","nav_open_cabinet","open_fridge_pick"]
    )
    # habitat_baselines.eval_keys_to_include_in_name: List[str]=field(
    #     default_factory=lambda:['composite_pddl_success','pick_obj_measure','is_art_measure','force_terminate','wrong_pick_measure','wrong_marker_pick_measure']
    # )
    eval_group_by: str = "is_art_measure"
    eval_group_measures: List[str] = field(
        default_factory=lambda:["composite_pddl_success", "pick_obj_measure"]
    )
    skills: SkillConfig = SkillConfig()

@dataclass
class MonoMetaHierSkillSensorConfig(LabSensorConfig):
    type: str = "MonoMetaHierSkillSensor"
    art_success_id: str = "art_obj_at_desired_state" 
    art_success_dist_threshold: float = 0.30
    allowed_skills: List[str] = field(default_factory=lambda:["nav_pick", "nav_place", "nav_open_cabinet", "nav_open_fridge", "nav_pick_nav_place"])

cs.store(group="habitat", name="habitat_config_base", node=MonoHabitatConfig)
cs.store(
    group="habitat.environment",
    name="environment_config_schema",
    node=MonoEnvironmentConfig,
)
cs.store(
    package="habitat.task",
    group="habitat/task",
    name="custom_task_config_base",
    node=CustomTaskConfig,
)

cs.store(
    package="habitat.task.measurements.env_task_measure",
    group="habitat/task/measurements",
    name="env_task_measure",
    node=CustomMonoMeasureConfig,
)

# Hier skill sensor config #
cs.store(
    package="habitat.task.lab_sensors.hier_skill_sensor",
    group="habitat/task/lab_sensors",
    name="hier_skill_sensor",
    node=MonoHierSkillSensorConfig,
)

# Meta hier skill sensor config #
cs.store(
    package="habitat.task.lab_sensors.meta_hier_skill_sensor",
    group="habitat/task/lab_sensors",
    name="meta_hier_skill_sensor",
    node=MonoMetaHierSkillSensorConfig,
)

# Arm Action config # 
cs.store(
    package="habitat.task.actions.arm_action",
    group="habitat/task/actions",
    name="arm_action",
    node=MonoArmActionConfig,
)

# Base Action config for the task # 
cs.store(
    package="habitat.task.actions.base_velocity",
    group="habitat/task/actions",
    name="base_velocity",
    node=MonoBaseVelocityActionConfig,
)

cs.store(
    package="habitat.task.measurements.art_obj_at_desired_state",
    group="habitat/task/measurements",
    name="art_obj_at_desired_state",
    node=MonoArtObjAtDesiredStateMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.task_stage_steps",
    group="habitat/task/measurements",
    name="task_stage_steps",
    node=MonoTaskStageStepsMeasurementConfig,
)

cs.store(
    package="habitat.task.lab_sensors.one_hot_target_sensor",
    group="habitat/task/lab_sensors",
    name="one_hot_target_sensor",
    node=OneHotTargetSensorConfig,
)
cs.store(
    package="habitat.task.measurements.ee_dist_to_marker",
    group="habitat/task/measurements",
    name="ee_dist_to_marker",
    node=MonoEndEffectorDistToMarkerConfig,
)

cs.store(
    package="habitat.task.measurements.move_objects_reward",
    group="habitat/task/measurements",
    name="move_objects_reward",
    node=MonoMoveObjectsRewardMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.diff_joint_velocity",
    group="habitat/task/measurements",
    name="diff_joint_velocity",
    node=DiffJointVelocityMeasurementConfig,
)

cs.store(
    group="env_tasks",
    name="env_tasks_config_base",
    node=EnvTasksConfig(),
)
# cs.store(
#     package="habitat.task.measurements.mono_allowed_skills",
#     group="habitat/task/measurements",
#     name="mono_allowed_skills",
#     node=MonoAllowedSkillsMeasurementConfig,
# )


cs.store(
    package="habitat.task.lab_sensors.goal_sensor",
    group="habitat/task/lab_sensors",
    name="goal_sensor",
    node=MonoGoalSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.target_start_sensor",
    group="habitat/task/lab_sensors",
    name="target_start_sensor",
    node=MonoTargetStartSensorConfig,
)
cs.store(
    package="habitat.task.measurements.composite_pddl_success",
    group="habitat/task/measurements",
    name="composite_pddl_success",
    node=CustomPddlSuccessConfig,
)

cs.store(
    package="habitat.task.measurements.obj_at_goal_pddl_success",
    group="habitat/task/measurements",
    name="obj_at_goal_pddl_success",
    node=ObjAtGoalPddlSuccessConfig,
)

cs.store(
    package="habitat.task.measurements.pick_marker_measure",
    group="habitat/task/measurements",
    name="pick_marker_measure",
    node=MonoDidPickMarkerMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.pick_obj_measure",
    group="habitat/task/measurements",
    name="pick_obj_measure",
    node=MonoDidPickObjectMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.pick_correct_obj_measure",
    group="habitat/task/measurements",
    name="pick_correct_obj_measure",
    node=MonoDidPickCorrectObjectMeasurementConfig,

)

cs.store(
    package="habitat.task.measurements.mono_pddl_stage_goals",
    group="habitat/task/measurements",
    name="mono_pddl_stage_goals",
    node=MonoPddlStageGoalsConfig,
)

cs.store(
    package="habitat.task.measurements.art_obj_state",
    group="habitat/task/measurements",
    name="art_obj_state",
    node=MonoArtObjStateMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.mono_nav_to_obj_success",
    group="habitat/task/measurements",
    name="mono_nav_to_obj_success",
    node=MonoNavToObjSuccessMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.did_nav_to_obj_success",
    group="habitat/task/measurements",
    name="did_nav_to_obj_success",
    node=DidMonoNavToObjSuccessMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.did_nav_to_pos_success",
    group="habitat/task/measurements",
    name="did_nav_to_pos_success",
    node=DidMonoNavToPosSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.did_open_art_receptacle",
    group="habitat/task/measurements",
    name="did_open_art_receptacle",
    node=DidOpenArticulatedReceptacle,
)

cs.store(
    package="habitat.task.measurements.art_obj_success",
    group="habitat/task/measurements",
    name="art_obj_success",
    node=MonoArtObjSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.art_success_pos",
    group="habitat/task/measurements",
    name="art_success_pos",
    node=MonoArtSuccessPosMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.art_obj_reward",
    group="habitat/task/measurements",
    name="art_obj_reward",
    node=MonoArtObjRewardMeasurementConfig,
)

# cs.store(
#     group="habitat_baselines",
#     name="habitat_baselines_rl_config_base",
#     node=CustomHabitatBaselinesRLConfig(),
# )

cs.store(
    package="habitat.task.measurements.is_art_measure",
    group="habitat/task/measurements",
    name="is_art_measure",
    node=IsArtMeasureConfig,
)

cs.store(
    package="habitat.task.measurements.all_rank_is_art_measure",
    group="habitat/task/measurements",
    name="all_rank_is_art_measure",
    node=AllRankIsArtMeasureConfig,
)
cs.store(
    package="habitat.task.measurements.mono_nav_to_pos_succ",
    group="habitat/task/measurements",
    name="mono_nav_to_pos_succ",
    node=MonoNavToPosSuccConfig,
)
cs.store(
    package="habitat.task.measurements.mono_rot_dist_to_goal",
    group="habitat/task/measurements",
    name="mono_rot_dist_to_goal",
    node=MonoRotDistToGoalConfig,
)

cs.store(
    package="habitat.task.measurements.end_effector_to_object_distance",
    group="habitat/task/measurements",
    name="end_effector_to_object_distance",
    node=MonoEndEffectorToObjectDistanceMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.object_to_goal_distance",
    group="habitat/task/measurements",
    name="object_to_goal_distance",
    node=MonoObjectToGoalDistanceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.end_effector_to_goal_distance",
    group="habitat/task/measurements",
    name="end_effector_to_goal_distance",
    node=MonoEndEffectorToGoalDistanceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.pick_reward",
    group="habitat/task/measurements",
    name="pick_reward",
    node=MonoRearrangePickRewardMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.pick_success",
    group="habitat/task/measurements",
    name="pick_success",
    node=MonoRearrangePickSuccessMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.mono_dist_to_goal",
    group="habitat/task/measurements",
    name="mono_dist_to_goal",
    node=MonoDistToGoalConfig,
)

cs.store(
    package="habitat.task.measurements.mono_nav_to_obj_reward",
    group="habitat/task/measurements",
    name="mono_nav_to_obj_reward",
    node=MonoNavToObjRewardConfig,
)


cs.store(
    package="habitat.task.lab_sensors.env_task_sensor",
    group="habitat/task/lab_sensors",
    name="env_task_sensor",
    node=EnvTaskSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.marker_rel_pos_sensor",
    group="habitat/task/lab_sensors",
    name="marker_rel_pos_sensor",
    node=MonoMarkerRelPosSensorConfig,
)
cs.store(
    package="habitat.task.measurements.wrong_pick_measure",
    group="habitat/task/measurements",
    name="wrong_pick_measure",
    node=WrongPickMeasureConfig,
)
cs.store(
    package="habitat.task.measurements.wrong_pick_marker_measure",
    group="habitat/task/measurements",
    name="wrong_pick_marker_measure",
    node=WrongMarkerPickMeasureConfig,
)
cs.store(
    package="habitat.task.lab_sensors.env_task_sensor",
    group="habitat/task/lab_sensors",
    name="custom_mono_sensor",
    node=CustomMonoSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.is_art_sensor", 
    group="habitat/task/lab_sensors",
    name="is_art_sensor",
    node=IsArtSensorConfig
)

# cs.store(
#     package="habitat.task.measurements.hier_skill_measure",
#     group="habitat/task/measurements",
#     name="hier_skill_measure",
#     node=MonoHierSkillMeasurementConfig,
# )

cs.store(
    package="habitat_baselines.wb",
    group="habitat_baselines/wb",
    name="custom_wb_config",
    node=CustomWeightsandBiasesConfig,
)
