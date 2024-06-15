#!/bin/bash
export task_name="rearrange"
export PROJ_NAME="skill-chain" 
bash scripts/base_cmd.sh \
env_tasks.allowed_skills=["nav_pick_nav_place"] \
habitat_baselines.evaluate=True \
habitat_baselines.load_resume_state_config=False \
habitat_baselines.num_environments=1 \
habitat.dataset.data_path="data/datasets/replica_cad/rearrange/v1/val/rearrange_hard_200.json.gz" \
habitat.task.measurements.force_terminate.max_instant_force=-1.0 \
habitat.task.measurements.force_terminate.max_accum_force=-1.0 \
env_tasks.skills.nav_pick_nav_place.update_measures.task.measurements.force_terminate.max_instant_force=-1.0 \
habitat_baselines.video_dir="data/videos/rearrange/" \
habitat_baselines.eval_ckpt_path_dir="data/ckpts/rearrange/latest.pth" \
habitat.gym.obs_keys=["obj_start_sensor","obj_goal_sensor","head_depth","joint","ee_pos","is_holding","relative_resting_position","env_task_id_sensor","is_art_sensor","hier_skill_sensor"] \
habitat_baselines.wb.run_name="rearrange_eval" "$@"