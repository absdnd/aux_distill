# Reinforcement Learning via Auxiliary Task Distillation

We present Reinforcement Learning via Auxiliary Task Distillation (AuxDistill),
a new method that enables reinforcement learning (RL) to perform long-horizon
robotic control problems by distilling behaviors from auxiliary RL tasks. AuxDistill
achieves this by concurrently carrying out multi-task RL with auxiliary tasks,
which are easier to learn and relevant to the main task. A weighted distillation
loss transfers behaviors from these auxiliary tasks to solve the main task. We
demonstrate that AuxDistill can learn a pixels-to-actions policy for a challenging
multi-stage embodied object rearrangement task from the environment reward
without demonstrations, a learning curriculum, or pre-trained skills. AuxDistill
achieves 2.3× higher success than the previous state-of-the-art baseline in the
Habitat Object Rearrangement benchmark and outperforms methods that use pre-
trained skills and expert demonstrations. 


# Installation 
- `mamba create -n hab_sc -y python=3.9`
- `mamba install -y habitat-sim==0.3.0 withbullet headless -c conda-forge -c aihabitat`
- In another directory:
    - `git clone https://github.com/facebookresearch/habitat-lab.git`
    - `git checkout tags/v0.3.1`
    - `cd habitat-lab`
    - `pip install -e habitat-lab`
    - `pip install -e habitat-baselines`
- `cd` back to project directory
- Download datasets: `python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets`

## Datasets 

To download the datasets used for training use the datasets provided in the `train` folder at the following [link](https://drive.google.com/drive/folders/1TBaCxWnNP8xYoluNl1Yrh7zWYS3CkJkA?usp=sharing) and place it in the `data/datasets/replica_cad/rearrange/train` folder. Place the validation dataset in the `data/datasets/replica_cad/rearrange/val` folder. 


## Training 
  ### Rearrangement 
  To train the code for performing the rearrangement task use - `bash skill_chain/train/rearrange.sh`. This will train the model using the configuration file `configs/rearrange.yaml`. The checkpoints will by default be saved to the `data/ckpts/rearrange ` directory. To train different seeds simply append the argument `habitat.seed=$SEED` to the command. 

  ### Language Pick  
  To execute the language pick task use the command - `bash skill_chain/train/lang_pick.sh`. To train a different seed, append `habitat.seed=$SEED` to the run command 


## Evaluation

### Pretrained checkpoints

The pre-trained checkpoints for the rearrangement and language pick tasks can be found at the following [link](https://drive.google.com/drive/folders/1rRP66q7G_DUquCaukx2On7ED_ZeNbIDM?usp=drive_link), place the checkpoints in the `data/ckpts/pretrained` folder. To run the pretrained checkpoint, run `bash scripts/eval/rearrange_pretrained.sh` to execute the run command. Change the checkpoint path to evaluate different seeds in this experiment. For language pick evaluation, run `bash scripts/eval/lang_pick_pretrained.sh` to evaluate the pretrained language pick model.

### Evaluating Trained Checkpoints
Use the command, `bash scripts/eval/rearrange.sh` to evaluate the trained checkpoints. By default this evaluates the latest checkpoint saved in `data/ckpts/rearrange/latest.pth`. This path can be adjusted in the evaluation script. The videos are generated in the path given by  `habitat_baselines.video_dir` defined in the script.  


## Documentation 

- `configs/:` Contains the config files used for running habitat. The path `config/aux_distill/rl_auxdistill.yaml` contains the task definitions used for both the rearrangement and language pick tasks. The config paths for the rearrangement and language pick tasks are defined in `config/aux_distill/rearrange.yaml` and `config/aux_distill/lang_pick.yaml` respectively.

- `pddl/`: Contains the PDDL task specification for the rearrangement at `rearrange_easy.yaml` and the `nav_pick.yaml` and the `nav_place.yaml` task. 

- `policy/aux_mtask_policy.py`: Contains the policy definition used for multi-task RL training. The class `DeepCriticHead` implements Pop-Art return normalization and the multi-task value head. 

- `tasks/` defines a custom version of the `PddlTask` and the language pick task used in our training setup. We also define variants of the habitat-pick, nav, and place task to support auxiliary task training. 

- `sensors/` contains the sensors used in the rearrangement task. This includes `mono_nav_to_obj_sensors.py` and `mono_sensors.py` contain the definition of the sensors used in the rearrangement task.

- `trainer/` : The file `trainer/aux_trainer.py` contains a variant of the main training loop, which supports stratified logging of metrics based on `easy` and `hard` tasks. `aux_ppo.py`implements the distillation loss which is used for training the policy. The `trainer/custom_gym_env.py` class implements a custom gym environment which supports the creation of a multi-task RL training scheme.



## Citation 

@article{harish2024,
  title={Reinforcement Learning via Auxiliary Task Distillation},
  author={Harish, Abhinav, Heck, Larry, Hanna Josiah, Kira Zsolt and Szot Andrew},
  journal={arXiv preprint arXiv:2303.16194},
  year={2023}
}