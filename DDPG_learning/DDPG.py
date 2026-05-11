import os
import torch
import torch.nn as nn
import torch.optim as optim

from ActorCritic import Actor, Critic

class DDPG:
    def __init__(self, state_dim, action_dim, max_action,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=1e-3,
                 device=None, save_dir=None, weight_decay_critic=1e-3):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.max_action = float(max_action)

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay_critic)

        self.mse = nn.MSELoss()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        proj_root = os.path.abspath(os.path.join(current_dir, ".."))
        self.save_dir = save_dir or os.path.join(proj_root, "DDPGLearnedModel")
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad()
    def select_action(self, state_np):
        state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(state).cpu().numpy().squeeze(0)
        return action

    def train_step(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return None
        
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
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.gamma * target_Q

        current_Q = self.critic(state, action)
        critic_loss = self.mse(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }

    def _soft_update(self, net, target_net):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def save(self, name="ddpg"):
        torch.save(self.actor.state_dict(),  os.path.join(self.save_dir, f"{name}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(self.save_dir, f"{name}_critic.pth"))

    def load(self, name="ddpg"):
        self.actor.load_state_dict(torch.load(os.path.join(self.save_dir, f"{name}_actor.pth"), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(self.save_dir, f"{name}_critic.pth"), map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
