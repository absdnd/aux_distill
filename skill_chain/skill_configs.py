from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Optional
from omegaconf import MISSING


@dataclass
class DistillTaskConfig():
    task_id: int = MISSING
    type: str = MISSING
    reward_measure: str = MISSING
    success_measure: str = MISSING
    env_task_gym_id: str = MISSING
    task_spec_base_path: str = MISSING
    task_spec: str = MISSING
    update_measures: Dict[str, Any] = MISSING
    update_rl_params: Dict[str, Any] = MISSING
    max_episode_steps: int = MISSING
    measures_list: List[str] = MISSING
    log_art_list: List[str] = field(default_factory=lambda: [])
    exclude_measures: List[str] = field(default_factory=lambda: [])
    use_marker_options: List[str] = field(default_factory=lambda: [])
    pddl_def: Dict[str, Any] = field(default_factory=lambda: {
        "start_template": [],
        "expr_type": "",
        "sample_entities": {},
        "goal_template": {},
    })
    pddl_domain_def:str = MISSING
    exclude_obs_keys: List[str] = field(default_factory=lambda: [])


