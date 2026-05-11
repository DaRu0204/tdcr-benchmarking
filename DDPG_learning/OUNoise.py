import numpy as np

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, dt=1.0, max_action=1.0):
        self.action_dim = action_dim
        
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        
        self.max_action = float(max_action)
        
        self.mu = np.full(action_dim, mu, dtype=float)
        
        self.state = np.zeros(action_dim, dtype=float)

    def reset(self):
        self.state.fill(0.0)

    def sample(self):
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.state = self.state + dx
        return np.clip(self.state, -self.max_action, self.max_action)
