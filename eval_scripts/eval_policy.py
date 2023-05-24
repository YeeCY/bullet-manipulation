import argparse
import os.path

import roboverse as rv
import numpy as np
import imageio

from gym.wrappers import ClipAction
from eval_scripts.images.renderer import EnvRenderer
from eval_scripts.images.insert_image_env import InsertImageEnv
from eval_scripts.contextual.goal_conditioned import PresampledPathDistribution
from eval_scripts.reward_fns import GoalReachingRewardFn
from eval_scripts.contextual.contextual_env import ContextualEnv
from eval_scripts.drawer_pnp_push_commands import drawer_pnp_push_commands

from eval_scripts.utils.io import load_local_or_remote_file
import eval_scripts.utils.pytorch_util as ptu
import rlkit.torch.pytorch_util as rlkit_ptu


def path_collector_obs_processor(
        obs,
        path_collector_observation_keys=['image_observation'],
        path_collector_context_keys_for_policy=['image_desired_goal'],
):
    combined_obs = []

    for k in path_collector_observation_keys:
        combined_obs.append(obs[k])

    for k in path_collector_context_keys_for_policy:
        combined_obs.append(obs[k])

    return np.concatenate(combined_obs, axis=0)


def main(args):
    ptu.set_gpu_mode(1)
    rlkit_ptu.set_gpu_mode(1)

    high_res_obs_imgs = np.zeros(
        (args.num_trajectories * args.num_timesteps, args.env_obs_img_dim, args.env_obs_img_dim, 3))
    high_res_goal_imgs = np.zeros(
        (args.num_trajectories, args.env_obs_img_dim, args.env_obs_img_dim, 3))

    pretrained_rl_path = os.path.expanduser(args.pretrained_rl_path)
    rl_model_dict = load_local_or_remote_file(pretrained_rl_path, map_location=ptu.device)
    policy = rl_model_dict['evaluation/policy'].to(ptu.device)

    presampled_goal_dir = os.path.expanduser(args.presampled_goal_dir)
    presampled_goals_path = os.path.join(
        presampled_goal_dir,
        'td_pnp_push_scripted_goals_timeoutk{}_seed{}.pkl'.format(
            args.goal_timeoutk, args.eval_seed))

    raw_env = rv.make(
        "SawyerRigAffordances-v6",
        gui=False,
        expl=True,
        env_obs_img_dim=args.env_obs_img_dim,
        obs_img_dim=args.obs_img_dim,
        test_env=True,
        test_env_command=drawer_pnp_push_commands[args.eval_seed],
        downsample=True,
    )
    state_env = ClipAction(raw_env)
    renderer = EnvRenderer(
        create_image_format='HWC',
        output_image_format='CWH',
        flatten_image=True,
        width=args.obs_img_dim,
        height=args.obs_img_dim)
    env = InsertImageEnv(
        state_env,
        renderer=renderer)

    diagnostics = env.get_contextual_diagnostics
    context_distribution = PresampledPathDistribution(
        presampled_goals_path,
        None,
        initialize_encodings=False)
    reward_fn = GoalReachingRewardFn(
        state_env.env,
        use_pretrained_reward_classifier_path=False,
        obs_type='state',
        reward_type='highlevel',
        epsilon=3.0,
        terminate_episode=0
    )

    env = ContextualEnv(
        env,
        context_distribution=context_distribution,
        reward_fn=reward_fn,
        observation_key='latent_observation',
        contextual_diagnostics_fns=[diagnostics] if not isinstance(
            diagnostics, list) else diagnostics,
    )

    traj_idx = 0
    while traj_idx < args.num_trajectories:
        print("Trajectory: {}".format(traj_idx))
        traj_obses, traj_goals = [], []

        while True:
            try:
                obs = env.reset()
                break
            except:
                print("Reset error, trying again.")

        for t in range(args.num_timesteps):
            # Render Image
            downsample = raw_env.downsample
            raw_env.downsample = False
            high_res_obs_img = np.uint8(env.render_obs())
            raw_env.downsample = downsample

            high_res_obs_imgs[traj_idx * args.num_timesteps + t, :] = high_res_obs_img

            obs = path_collector_obs_processor(obs)
            action, _ = policy.get_action(obs)
            obs, reward, _, info = env.step(action)

            traj_obses.append(obs['state_observation'])
            traj_goals.append(obs['state_desired_goal'])

        traj_obses = np.array(traj_obses)
        traj_goals = np.array(traj_goals)
        success = env.get_success_metric(traj_obses, traj_goals, key='overall')
        final_success = success[-1]

        if final_success:
            high_res_goal_imgs[traj_idx] = high_res_obs_img.copy()
            traj_idx += 1
            print("Trajectory {} succeed!".format(traj_idx))

    high_res_goal_imgs = np.repeat(high_res_goal_imgs[:, None], args.num_timesteps, axis=1)
    high_res_goal_imgs = high_res_goal_imgs.reshape([
        args.num_trajectories * args.num_timesteps, args.env_obs_img_dim, args.env_obs_img_dim, 3])
    video_imgs = np.concatenate([
        high_res_obs_imgs, high_res_goal_imgs], axis=2)

    video_save_dir = os.path.expanduser(args.video_save_dir)
    os.makedirs(video_save_dir, exist_ok=True)
    video_save_path = os.path.join(video_save_dir, args.video_filename + ".mp4")
    imageio.mimsave(video_save_path, video_imgs, fps=20)
    print("Video saved to: {}".format(os.path.abspath(video_save_path)))


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_trajectories', type=int, default=4)
    parser.add_argument('--num_timesteps', type=int, default=200)
    parser.add_argument('--video_save_dir', type=str, default='~/offline_c_learning/railrl_logs/eval_policy/videos')
    parser.add_argument('--video_filename', type=str, default='debug')
    parser.add_argument('--pretrained_rl_path', type=str,
                        default='~/offline_c_learning/railrl_logs/env6/aug_5_contrastive_nce_geom_future_goals_jax_hyperparams/31/0/pretrained/run0/id0/itr_300.pt')
    parser.add_argument('--presampled_goal_dir', type=str,
                        default='~/offline_c_learning/dataset/env6_td_pnp_push/goals_early_stop')
    parser.add_argument('--eval_seed', type=int, default=31)
    parser.add_argument('--goal_timeoutk', type=str, default=-1)
    parser.add_argument('--env_obs_img_dim', type=int, default=196)
    parser.add_argument('--obs_img_dim', type=int, default=48)

    args = parser.parse_args()
    main(args)
