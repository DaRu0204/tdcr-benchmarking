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

from ReplayBuffer import ReplayBuffer
from DQN import DQN
from ContinuumRobot import ContinuumRobotEnv
from DiscreteAction import DiscreteActionEnv

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


def run_single_experiment(
    state_type: int,
    include_u_abs: bool,
    reward_type: int,
    use_bonus_penalty: bool,
    total_episodes: int = 1500,
):
    """Train one DQN algorithm-configuration pair from the 24-cell benchmark grid."""
    set_global_seed(GLOBAL_SEED)

    u_flag = int(include_u_abs)
    b_flag = int(use_bonus_penalty)

    base_name = f"dqn_continuum_robot_s{state_type}u{u_flag}r{reward_type}b{b_flag}"

    step_penalty, success_bonus = get_reward_params(reward_type)

    use_success_bonus = use_bonus_penalty
    use_step_penalty = use_bonus_penalty

    base_env = ContinuumRobotEnv(
        max_action=1.0,
        reward_type=reward_type,
        use_success_bonus=use_success_bonus,
        success_bonus=success_bonus,
        use_step_penalty=use_step_penalty,
        step_penalty=step_penalty,
        state_type=state_type,
        include_u_abs=include_u_abs,
    )
    env = DiscreteActionEnv(base_env)

    agent = DQN(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
    )

    buffer = ReplayBuffer(capacity=1_000_000)

    batch_size = 128
    episode_rewards = deque(maxlen=100)
    episode_distances = deque(maxlen=100)
    times = []
    episode_steps = []
    successes_total = 0
    last_action_u = None

    if USE_WANDB:
        wandb.init(
            project="DQN_testing",
            name=base_name,
            config={
                "config_name": base_name,
                "device": str(agent.device),
                "cuda_available": torch.cuda.is_available(),
                "learning_rate": agent.lr,
                "episodes": total_episodes,
                "batch_size": batch_size,
                "gamma": agent.gamma,
                "tau": agent.tau,
                "eps_start": agent.eps_start,
                "eps_end": agent.eps_end,
                "eps_decay_episodes": agent.eps_decay_episodes,
                "env_max_step": ContinuumRobotEnv.max_step,
                "state_dim": env.state_dim,
                "n_actions": env.n_actions,
                "u_max": base_env.u_max,
                "max_action_delta": base_env.max_action,
                "state_type": state_type,
                "include_u_abs": include_u_abs,
                "reward_type": reward_type,
                "use_success_bonus": use_success_bonus,
                "success_bonus": success_bonus,
                "use_step_penalty": use_step_penalty,
                "step_penalty": step_penalty,
            },
        )

    log_file = f"{base_name}.csv"
    need_header = (not os.path.exists(log_file)) or (os.path.getsize(log_file) == 0)
    if need_header:
        with open(log_file, "w") as f:
            f.write(
                "episode,distance,episode_reward,avg_distance,avg_reward,"
                "episode_time,episode_step,success_rate,epsilon\n"
            )

    print(f"\n=== Starting experiment: {base_name} ===")
    print(
        f"state_type={state_type}, include_u_abs={include_u_abs}, "
        f"reward_type={reward_type}, use_bonus_penalty={use_bonus_penalty}"
    )

    for episode in range(total_episodes):
        ep_start = time.perf_counter()

        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action_idx = agent.select_action(state)

            next_state, reward, done, truncated = env.step(action_idx)

            buffer.add(
                state,
                action_idx,
                next_state,
                reward,
                float(done),
                float(truncated),
            )

            agent.train_step(buffer, batch_size=batch_size)

            state = next_state
            episode_reward += reward
            last_action_u = env.u

        ep_time = time.perf_counter() - ep_start

        dis = env.distance()
        times.append(ep_time)
        episode_rewards.append(episode_reward)
        episode_distances.append(dis)

        avg_reward = float(np.mean(episode_rewards))
        avg_distance = float(np.mean(episode_distances))

        ep_steps = base_env.current_step
        episode_steps.append(ep_steps)

        if dis < base_env.goal_eps:
            successes_total += 1
        success_rate_overall = successes_total / float(episode + 1)

        agent.decay_epsilon()

        print(
            f"[{base_name}] Ep {episode+1:5d} | R={episode_reward:9.3f} | "
            f"avg_r={avg_reward:9.3f} | dist={dis:.6f} | "
            f"avg_dist={avg_distance:.6f} | succ={success_rate_overall:.3f} | "
            f"steps={ep_steps} | {ep_time:.2f}s"
        )
        print(
            f"[{base_name}] Last action: {last_action_u} | "
            f"buffer step: {buffer.iterations}"
        )

        if USE_WANDB:
            log_payload = {
                "episode": episode + 1,
                "distance": dis,
                "episode_reward": episode_reward,
                "avg_distance": avg_distance,
                "avg_reward": avg_reward,
                "episode_time": ep_time,
                "episode_step": ep_steps,
                "success_rate": success_rate_overall,
                "epsilon": agent.epsilon,
            }
            wandb.log(log_payload)

        with open(log_file, "a") as f:
            f.write(
                f"{episode + 1},{dis:.6f},{episode_reward:.6f},"
                f"{avg_distance:.6f},{avg_reward:.6f},{ep_time:.6f},"
                f"{ep_steps},{success_rate_overall:.6f},{agent.epsilon:.6f}\n"
            )

        if (episode + 1) % 100 == 0:
            save_name = f"{base_name}_e{episode + 1}"
            agent.save(save_name)

    if len(times) > 0 and len(episode_steps) > 0:
        avg_episode_time = float(np.mean(times))
        avg_episode_steps = float(np.mean(episode_steps))
        with open(log_file, "a") as f:
            f.write(
                f"overall_avg,,,,,{avg_episode_time:.6f},{avg_episode_steps:.6f},,\n"
            )

    final_save_name = f"{base_name}_e{total_episodes}"
    agent.save(final_save_name)

    if USE_WANDB:
        wandb.finish()

    print(f"=== Finished experiment: {base_name} ===\n")


def main():
    total_episodes = 1500

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
                        total_episodes=total_episodes,
                    )


if __name__ == "__main__":
    curr = os.path.basename(os.getcwd())
    if curr != "DQN_learning":
        print("Tip: run this script from the DQN_learning directory for consistent paths.")
    main()
