# Installation

- `mamba create -n hab_mt -y python=3.9`
- `mamba install -y habitat-sim withbullet  headless -c conda-forge -c aihabitat`
- In another directory:
    - `git clone https://github.com/facebookresearch/habitat-lab.git`
    - `cd habitat-lab`
    - `pip install -e habitat-lab`
    - `pip install -e habitat-baselines`
- `cd` back to project directory
- Download datasets: `python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets`

# Running
- Typical run command `python run.py --config-name=baseline/rl.yaml habitat_baselines.wb.entity=$WB_ENTITY habitat_baselines.wb.run_name=$JOB_ID habitat_baselines.wb.project_name=$PROJECT_NAME habitat_baselines.checkpoint_folder=$DATA_DIR/checkpoints/$JOB_ID/ habitat_baselines.video_dir=$DATA_DIR/vids/$JOB_ID/ habitat_baselines.log_file=$DATA_DIR/logs/$JOB_ID.log habitat_baselines.tensorboard_dir=$DATA_DIR/tb/$JOB_ID/ habitat_baselines.writer_type=wb` where:
    - `$JOB_ID`: a unique job identifier
    - `$DATA_DIR`: base data directory. For example, `/srv/share/aszot3/habitat2`.
    - `$WB_ENTITY` your wandb user or team name.
    - `$PROJECT_NAME`: project name on wandb.



