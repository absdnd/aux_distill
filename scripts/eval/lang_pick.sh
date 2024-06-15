export PROJ_NAME="skill-chain"
export task_name="lang_pick"
bash scripts/base_cmd.sh \
habitat/simulator/sensor_setups@habitat.simulator.agents.main_agent=rgb_head_agent \
habitat.task.lab_sensors.env_task_sensor.max_skills=15 \
env_tasks.allowed_skills=["lang_pick_prox"] \
habitat_baselines.eval_keys_to_include_in_name=["pick_correct_obj_measure"] \
habitat_baselines.evaluate=True \
habitat_baselines.load_resume_state_config=False \
habitat_baselines.num_environments=1 \
habitat.dataset.data_path="data/datasets/replica_cad/rearrange/v1/val/rearrange_easy.json.gz" \
habitat.task.measurements.force_terminate.max_instant_force=-1.0 \
habitat.task.measurements.force_terminate.max_accum_force=-1.0 \
env_tasks.skills.nav_pick_nav_place.update_measures.task.measurements.force_terminate.max_instant_force=-1.0 \
habitat_baselines.video_dir="data/videos/lang_pick/" \
habitat_baselines.eval_ckpt_path_dir="data/ckpts/lang_pick/latest.pth" \
habitat.gym.obs_keys=["one_hot_target_sensor","obj_start_sensor","head_rgb","joint","is_holding","relative_resting_position","env_task_id_sensor","hier_skill_sensor"] "$@"
