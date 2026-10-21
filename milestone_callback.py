import os
from stable_baselines3.common.callbacks import BaseCallback


class MilestoneCallback(BaseCallback):
    def __init__(self, milestones, save_path, verbose=0):
        super().__init__(verbose)
        self.milestones = milestones
        self.save_path = save_path     
        self.iteration = 0

    def _on_training_start(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _on_rollout_start(self):
        if self.iteration in self.milestones:
            path = os.path.join(self.save_path, f"iter_{self.iteration}")
            self.model.save(path)
            if self.verbose:
                print(f"Saved milestone {path} at {self.num_timesteps} clicks")
        self.iteration += 1

    def _on_step(self):
        return True
