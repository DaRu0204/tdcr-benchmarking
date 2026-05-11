import os
import sys
import time
from collections import deque
import random
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ContinuumRobot import ContinuumRobotEnv
from PPO import PPO, PPOConfig

USE_WANDB = False
if USE_WANDB:
    import wandb

GLOBAL_SEED = 666


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_reward_params(reward_type: int):
    """Table 1 composite-reward parameters for the selected reward family."""
    if reward_type == 1:
        return 0.05, 5.0
    elif reward_type == 2:
        return 0.6, 60.0
    elif reward_type == 3:
        return 1.1, 100.0
    else:
        raise ValueError(f"Unsupported reward_type: {reward_type}")


@torch.no_grad()
def evaluate_episode(env: ContinuumRobotEnv, agent: PPO, deterministic: bool = True):
    obs = env.reset()
    done = False
    ep_reward = 0.0
    while not done:
        action_t, _, _ = agent.ac.act(obs, deterministic=deterministic)
        action = action_t.detach().cpu().numpy()
        obs, reward, done, truncated = env.step(action)
        ep_reward += float(reward)
    dist = float(env.distance())
    steps = int(env.current_step)
    success = bool(dist < env.goal_eps)
    return ep_reward, dist, steps, success


def collect_rollout_with_training_episode_stats(env: ContinuumRobotEnv, agent: PPO):
    """PPO logs update-level rows, so keep completed episode metrics from each rollout."""
    agent.buf.reset()

    obs = env.state if hasattr(env, "state") else env.reset()

    bootstrap_obs = obs
    last_done = False
    last_truncated = False

    train_ep_rewards = []
    train_ep_distances = []
    train_ep_steps = []

    ep_reward = 0.0

    for _ in range(agent.cfg.rollout_steps):
        with torch.no_grad():
            action_t, logp_t, value_t = agent.ac.act(obs, deterministic=False)
        action_np = action_t.detach().cpu().numpy()

        next_obs, reward, done, truncated = env.step(action_np)

        agent.buf.add(
            obs=obs,
            action=action_np,
            reward=float(reward),
            done=bool(done),
            truncated=bool(truncated),
            value=float(value_t.item()),
            logprob=float(logp_t.item()),
        )

        ep_reward += float(reward)

        bootstrap_obs = next_obs
        last_done = bool(done)
        last_truncated = bool(truncated)

        obs = next_obs

        if done:
            train_ep_rewards.append(ep_reward)
            train_ep_distances.append(float(env.distance()))
            train_ep_steps.append(int(env.current_step))

            obs = env.reset()
            ep_reward = 0.0

    # Bootstrap from the observation after the final stored transition.
    with torch.no_grad():
        last_obs_t = torch.as_tensor(bootstrap_obs, dtype=torch.float32, device=agent.device)
        last_value = agent.ac.critic(last_obs_t).squeeze(-1).item()

    agent.buf.compute_returns_and_advantages(
        last_value=last_value,
        last_done=last_done,
        last_truncated=last_truncated,
    )

    return agent.cfg.rollout_steps, train_ep_rewards, train_ep_distances, train_ep_steps


def run_single_experiment(
    state_type: int,
    include_u_abs: bool,
    reward_type: int,
    use_bonus_penalty: bool,
    total_updates: int = 1500,
):
    """Train one PPO configuration; CSV `episode` is the update index."""

    set_global_seed(GLOBAL_SEED)

    u_flag = int(include_u_abs)
    b_flag = int(use_bonus_penalty)
    base_name = f"ppo_continuum_robot_s{state_type}u{u_flag}r{reward_type}b{b_flag}"

    step_penalty, success_bonus = get_reward_params(reward_type)
    use_success_bonus = use_bonus_penalty
    use_step_penalty = use_bonus_penalty

    env = ContinuumRobotEnv(
        max_action=1.0,
        reward_type=reward_type,
        use_success_bonus=use_success_bonus,
        success_bonus=success_bonus,
        use_step_penalty=use_step_penalty,
        step_penalty=step_penalty,
        state_type=state_type,
        include_u_abs=include_u_abs,
    )

    cfg = PPOConfig(
        gamma=0.995,
        gae_lambda=0.90,
        clip_coef=0.3,
        vf_clip_coef=0.2,
        ent_coef=0.02,
        vf_coef=0.25,
        max_grad_norm=0.5,
        target_kl=0.02,

        learning_rate=1e-4,
        epochs=10,
        minibatch_size=128,
        anneal_lr=True,

        rollout_steps=1024,
        device=None,
        seed=GLOBAL_SEED,
    )

    actor_critic_kwargs = dict(
        hidden_sizes=(128, 256, 256, 128),
        activation=torch.nn.Tanh,
        ortho_init=True,
        init_log_std=0.5,
        log_std_bounds=(-1.5, 2.0),
    )

    agent = PPO(
        obs_dim=env.state_dim,
        act_dim=env.action_dim,
        max_action=env.max_action,
        config=cfg,
        actor_critic_kwargs=actor_critic_kwargs,
    )

    episode_rewards = deque(maxlen=100)
    episode_distances = deque(maxlen=100)
    times = []
    episode_steps = []
    successes_total = 0
    total_train_eps = 0

    if USE_WANDB:
        wandb.init(
            project="PPO_testing",
            name=base_name,
            config={
                "config_name": base_name,
                "device": str(agent.device),
                "cuda_available": torch.cuda.is_available(),
                "total_updates": total_updates,
                "rollout_steps": cfg.rollout_steps,
                "mini_batch_size": cfg.minibatch_size,
                "update_epochs": cfg.epochs,
                "gamma": cfg.gamma,
                "gae_lambda": cfg.gae_lambda,
                "lr": cfg.learning_rate,
                "clip_range": cfg.clip_coef,
                "vf_coef": cfg.vf_coef,
                "ent_coef": cfg.ent_coef,
                "max_grad_norm": cfg.max_grad_norm,
                "target_kl": cfg.target_kl,
                "lr_anneal": cfg.anneal_lr,
                "env_max_step": ContinuumRobotEnv.max_step,
                "state_dim": env.state_dim,
                "action_dim": env.action_dim,
                "u_max": env.u_max,
                "max_action_delta": env.max_action,
                "state_type": state_type,
                "include_u_abs": include_u_abs,
                "reward_type": reward_type,
                "use_success_bonus": use_success_bonus,
                "success_bonus": success_bonus,
                "use_step_penalty": use_step_penalty,
                "step_penalty": step_penalty,
                "hidden_sizes": list(actor_critic_kwargs["hidden_sizes"]),
                "activation": "Tanh",
            },
        )

    log_file = f"{base_name}.csv"
    need_header = (not os.path.exists(log_file)) or (os.path.getsize(log_file) == 0)
    if need_header:
        with open(log_file, "w") as f:
            f.write("episode,distance,episode_reward,avg_distance,avg_reward,episode_time,episode_step,success_rate\n")

    print(f"\n=== Starting experiment: {base_name} ===")
    print(
        f"state_type={state_type}, include_u_abs={include_u_abs}, "
        f"reward_type={reward_type}, use_bonus_penalty={use_bonus_penalty}"
    )

    for update in range(1, total_updates + 1):
        up_start = time.perf_counter()

        _, train_ep_rewards, train_ep_distances, train_ep_steps = collect_rollout_with_training_episode_stats(env, agent)

        for r, d in zip(train_ep_rewards, train_ep_distances):
            episode_rewards.append(float(r))
            episode_distances.append(float(d))
            total_train_eps += 1
            if float(d) < env.goal_eps:
                successes_total += 1

        agent.update()

        # If no episode ended during rollout, fall back to current distance and 0 reward for this update slice.
        if len(train_ep_rewards) > 0:
            train_reward_this_update = float(np.mean(train_ep_rewards))
            train_distance_this_update = float(np.mean(train_ep_distances))
            train_steps_this_update = int(np.mean(train_ep_steps))
        else:
            train_reward_this_update = 0.0
            train_distance_this_update = float(env.distance())
            train_steps_this_update = int(getattr(env, "current_step", 0))

        avg_reward = float(np.mean(episode_rewards)) if len(episode_rewards) > 0 else 0.0
        avg_distance = float(np.mean(episode_distances)) if len(episode_distances) > 0 else float("nan")
        success_rate_overall = (successes_total / float(total_train_eps)) if total_train_eps > 0 else 0.0

        eval_reward, eval_dist, eval_steps, eval_success = evaluate_episode(env, agent, deterministic=True)

        ep_time = time.perf_counter() - up_start
        times.append(ep_time)
        episode_steps.append(train_steps_this_update)

        print(
            f"[{base_name}] Up {update:5d} | R={train_reward_this_update:9.3f} | "
            f"avg_r={avg_reward:9.3f} | dist={train_distance_this_update:.6f} | "
            f"avg_dist={avg_distance:.6f} | succ={success_rate_overall:.3f} | "
            f"steps={train_steps_this_update} | {ep_time:.2f}s"
        )
        print(
            f"[{base_name}] Eval | R={eval_reward:9.3f} | dist={eval_dist:.6f} | steps={eval_steps} | "
            f"success={int(eval_success)}"
        )

        with open(log_file, "a") as f:
            f.write(
                f"{update},{train_distance_this_update:.6f},{train_reward_this_update:.6f},"
                f"{avg_distance:.6f},{avg_reward:.6f},{ep_time:.6f},{train_steps_this_update},"
                f"{success_rate_overall:.6f}\n"
            )

        if USE_WANDB:
            log_payload = {
                "episode": update,
                "distance": train_distance_this_update,
                "episode_reward": train_reward_this_update,
                "avg_distance": avg_distance,
                "avg_reward": avg_reward,
                "episode_time": ep_time,
                "episode_step": train_steps_this_update,
                "success_rate": success_rate_overall,
            }
            wandb.log(log_payload)

        if update % 100 == 0:
            save_name = f"{base_name}_e{update}"
            agent.save(os.path.join(PROJECT_ROOT, "PPOLearnedModel", f"{save_name}.pth"))

    if len(times) > 0 and len(episode_steps) > 0:
        avg_episode_time = float(np.mean(times))
        avg_episode_steps = float(np.mean(episode_steps))
        with open(log_file, "a") as f:
            f.write(f"overall_avg,,,,,{avg_episode_time:.6f},{avg_episode_steps:.6f},\n")

    final_save_name = f"{base_name}_e{total_updates}"
    agent.save(os.path.join(PROJECT_ROOT, "PPOLearnedModel", f"{final_save_name}.pth"))

    if USE_WANDB:
        wandb.finish()

    print(f"=== Finished experiment: {base_name} ===\n")


def main():
    total_updates = 1500

    # State family x actuator augmentation x reward family x composite augmentation.
    for state_type in [1, 2]:
        for include_u_abs in [False, True]:
            for reward_type in [1, 2, 3]:
                for use_bonus_penalty in [False, True]:
                    run_single_experiment(
                        state_type=state_type,
                        include_u_abs=include_u_abs,
                        reward_type=reward_type,
                        use_bonus_penalty=use_bonus_penalty,
                        total_updates=total_updates,
                    )


if __name__ == "__main__":
    curr = os.path.basename(os.getcwd())
    if curr != "PPO_learning":
        print("Tip: run this script from the PPO_learning directory for consistent paths.")
    main()
