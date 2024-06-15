
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.single_agent_access_mgr import SingleAgentAccessMgr
from typing import Dict

@baseline_registry.register_agent_access_mgr
class MonoAgentAccessMgr(SingleAgentAccessMgr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Avoiding a broken critic for evaluatin 
    def load_state_dict(self, state: Dict) -> None:

        updated_state = {}
        for param_name, param_v in state["state_dict"].items():
            if "critic" not in param_name:
                updated_state[param_name] = param_v

        self._actor_critic.load_state_dict(updated_state, strict=False)
        if self._updater is not None:
            self._updater.load_state_dict(state)
            if "lr_sched_state" in state:
                self._lr_scheduler.load_state_dict(state["lr_sched_state"])


