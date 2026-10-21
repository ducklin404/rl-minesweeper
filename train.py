from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from minesweeper_env import MineEnv
from featureCNN import FeatureCNN



env = MineEnv()
maskEnv = ActionMasker(env, lambda env: env.action_masks())

policy_kwargs = {
    "features_extractor_class": FeatureCNN,
    "features_extractor_kwargs": {
        "features_dim": 256
    }
}


model = MaskablePPO(policy="MlpPolicy", policy_kwargs=policy_kwargs, env=maskEnv, verbose=1)
print(model.policy)
model.learn(total_timesteps=50_000)
model.save("minesweeper_cnn")