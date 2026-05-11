import os
import torch
import torch.nn as nn
import torch.optim as optim

from ActorCritic import Actor, Critic

class TD3:
    def __init__(self, state_dim, action_dim, max_action,
        lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, tau=0.005,
        policy_noise=0.2, noise_clip=0.5, policy_delay=2, device=None,
        save_dir=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.max_action = float(max_action)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = int(policy_delay)
        self.lr_actor = float(lr_actor)
        self.lr_critic = float(lr_critic)

        self.actor = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        self.mse = nn.MSELoss()
        self.total_it = 0

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        proj_root = os.path.abspath(os.path.join(curr_dir, ".."))
        self.save_dir = save_dir or os.path.join(proj_root, "TD3LearnedModel")
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad()
    def select_action(self, state_np):
        s = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(s).cpu().numpy().squeeze(0)
        return a

    def train_step(self, replay_buffer, batch_size=100):
        if len(replay_buffer) < batch_size:
            return None

        self.total_it += 1

        state, action, next_state, reward, done, truncated = replay_buffer.sample(batch_size)
        state      = state.to(self.device)
        action     = action.to(self.device)
        next_state = next_state.to(self.device)
        reward     = reward.to(self.device)
        done       = done.to(self.device)
        truncated  = truncated.to(self.device)

        # Time-limit truncations are bootstrapped, not treated as true terminals.
        not_done = 1.0 - (done * (1.0 - truncated))

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = reward + not_done * self.gamma * torch.min(target_q1, target_q2)

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        loss_q1 = self.mse(current_q1, target_q)
        loss_q2 = self.mse(current_q2, target_q)

        self.critic1_opt.zero_grad()
        loss_q1.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        loss_q2.backward()
        self.critic2_opt.step()

        info = {"critic1_loss": loss_q1.item(), "critic2_loss": loss_q2.item()}

        if replay_buffer.iterations % self.policy_delay == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            info["actor_loss"] = actor_loss.item()

            self._soft_update(self.actor,   self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        return info

    def _soft_update(self, net, target_net):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def save(self, name="td3"):
        torch.save(self.actor.state_dict(),   os.path.join(self.save_dir, f"{name}_actor.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(self.save_dir, f"{name}_critic1.pth"))
        torch.save(self.critic2.state_dict(), os.path.join(self.save_dir, f"{name}_critic2.pth"))

    def load(self, name="td3"):
        self.actor.load_state_dict(  torch.load(os.path.join(self.save_dir, f"{name}_actor.pth"),   map_location=self.device))
        self.critic1.load_state_dict(torch.load(os.path.join(self.save_dir, f"{name}_critic1.pth"), map_location=self.device))
        self.critic2.load_state_dict(torch.load(os.path.join(self.save_dir, f"{name}_critic2.pth"), map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
