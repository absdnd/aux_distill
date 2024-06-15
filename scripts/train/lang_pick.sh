#!/bin/bash
export PROJ_NAME="skill-chain"
export task_name="lang_pick"
bash scripts/base_cmd.sh \
env_tasks.allowed_skills=["pick","lang_pick_prox"] \
habitat_baselines.num_environments=14 \
habitat_baselines.total_num_steps=3.0e8 \
habitat_baselines.num_checkpoints=30 \
habitat.task.lab_sensors.env_task_sensor.max_skills=15 \
habitat/simulator/sensor_setups@habitat.simulator.agents.main_agent=rgb_head_agent \
habitat.task.measurements.task_stage_steps.force_early_end=False \
+habitat_baselines.rl.ppo.dist_skill_id=14 \
habitat.task.lab_sensors.env_task_sensor.max_skills=15 \
habitat.gym.obs_keys=["one_hot_target_sensor","obj_start_sensor","head_rgb","joint","is_holding","relative_resting_position","env_task_id_sensor","hier_skill_sensor"] "$@" 
