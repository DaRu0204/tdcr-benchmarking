import numpy as np

class GaussianNoise:
    def __init__(self, action_dim: int, max_action: float, std: float = 0.1):
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.std = float(std)

    def add(self, action):
        noise = np.random.normal(0.0, self.std, size=self.action_dim)
        return np.clip(np.asarray(action, dtype=float) + noise, -self.max_action, self.max_action)

    def add_noise(self, action):
        return self.add(action)
