# bullet-manipulation

This repo contains PyBullet-based manipulation environments consisting of a Sawyer robot manipulating drawers and picking and placing objects.

These environments are featured in:

[What Can I Do Here? Learning New Skills by Imagining Visual Affordances](https://arxiv.org/abs/2106.00671)
Alexander Khazatsky*, Ashvin Nair*, Daniel Jing, Sergey Levine. International Conference on Robotics and Automation (ICRA), 2021.

[Offline Meta-Reinforcement Learning with Online Self-Supervision](https://arxiv.org/abs/2107.03974)
Vitchyr H. Pong, Ashvin Nair, Laura Smith, Catherine Huang, Sergey Levine. arXiv preprint, 2021.

This repository extends https://github.com/avisingh599/roboverse which was developed by Avi Singh, Albert Yu, Jonathan Yang, Michael Janner, Huihan Liu, and Gaoyue Zhou. If you want to use ShapeNet objects, please download it from that repository: https://github.com/avisingh599/roboverse/tree/master/roboverse/assets/bullet-objects

## Setup

- install packages

  ```pip install -r requirements.txt```

- Clone repository `multiworld` from [here](https://github.com/vitchyr/multiworld.git).

- Create `PYTHONPATH` environment variable, replace `PYATHON_TO_RAILRL_PRIVATE_REPO`, `PATH_TO_THIS_REPO`, and `PATH_TO_MULTIWORLD_REPO`.

  ```
  export PYTHONPATH=$PYTHONPATH:PYATHON_TO_RAILRL_PRIVATE_REPO
  export PYTHONPATH=$PYTHONPATH:PATH_TO_THIS_REPO/roboverse/envs/assets/bullet-objects
  export PYTHONPATH=$PYTHONPATH:PATH_TO_MULTIWORLD_REPO
  ```

## Dataset Collection

```
python shapenet_scripts/env6_demo_collector_target.py --save_path SAVE_PATH --num_timesteps 75 --reset_interval 4 --num_trajectories_per_task_per_setting 200 --num_threads NUM_THREADS --num_tasks NUM_TASKS
```

Set NUM_TASKS=75 will collect a dataset with around 1.1M transitions.

## Pre-sampled Goals

Pre-sampled goals are contained in directory `goals_early_stop`.

## Policy Evaluation

A script is provided in `eval_scripts/eval_policy.py`. It was adapted from `rlkit` and is supposed to run with `railrl-private` repo.

The success metric can be computed using `get_success_metric` function like [here](https://github.com/YeeCY/bullet-manipulation/blob/7e02ece9e247f3aaf013abf09d88dd141a7c0424/eval_scripts/eval_policy.py#L130).
