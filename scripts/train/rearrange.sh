export task_name="rearrange"
export PROJ_NAME="skill-chain" # Replace with your project name
bash scripts/base_cmd.sh \
habitat_baselines.evaluate=False \
habitat.dataset.data_path="data/datasets/replica_cad/rearrange/v1/train/easy_hard.json.gz" \
habitat_baselines.checkpoint_folder="data/ckpts/rearrange/" \
habitat_baselines.log_file="data/logs/rearrange.log" \
habitat_baselines.wb.run_name="rearrange" \
+habitat_baselines.rl.ppo.dist_skill_id=5 \
habitat.gym.obs_keys=["obj_start_sensor","obj_goal_sensor","head_depth","joint","ee_pos","is_holding","relative_resting_position","env_task_id_sensor","is_art_sensor","hier_skill_sensor"] "$@"

