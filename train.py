import os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from minesweeper_env import MineEnv
from minePolicy import MinePolicy, ConvTrunk
from milestone_callback import MilestoneCallback


N_ENVS = 4
TOTAL_TIMESTEPS = 20_000_000
RUN = "cnn_v1"

# /kaggle/working on Kaggle, ./runs on the laptop
BASE_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "runs"
SAVE_DIR = os.path.join(BASE_DIR, RUN)


def make_env():
    env = MineEnv()
    env = ActionMasker(env, lambda env: env.action_masks())
    return env


if __name__ == "__main__":

    train_env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    eval_env = DummyVecEnv([lambda: Monitor(make_env())])

    # stop = StopTrainingOnNoModelImprovement(
    #     max_no_improvement_evals=10,
    #     min_evals=10,
    #     verbose=1,
    # )

    eval_cb = MaskableEvalCallback(
        eval_env,
        eval_freq=50_000 // N_ENVS,
        n_eval_episodes=200,
        best_model_save_path=os.path.join(SAVE_DIR, "best"),
        log_path=os.path.join(SAVE_DIR, "eval_logs"),
        # callback_after_eval=stop,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=500_000 // N_ENVS,
        save_path=os.path.join(SAVE_DIR, "checkpoints"),
        name_prefix="ckpt",
        verbose=1,
    )

    milestone_cb = MilestoneCallback(
        milestones=[0, 1, 10, 50, 100, 500, 1000, 2000],
        save_path=os.path.join(SAVE_DIR, "milestones"),
        verbose=1,
    )

    policy_kwargs = {
        "features_extractor_class": ConvTrunk,
        "normalize_images": False
    }

    model = MaskablePPO(policy=MinePolicy,
                        policy_kwargs=policy_kwargs,
                        env=train_env,
                        verbose=1,
                        batch_size=1024,
                        gamma=0.95,
                        ent_coef=0.01)
    print(model.policy)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, 
                callback=[eval_cb, checkpoint_cb, milestone_cb]
                )
    model.save(os.path.join(SAVE_DIR, "final_model"))
