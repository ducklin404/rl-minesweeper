# rl-minesweeper

An RL agent that learns to play Minesweeper. Built for the YouTube video "AI trains to win Minesweeper".

Stack: NumPy game engine → Gymnasium env → MaskablePPO (sb3-contrib) with a convolutional policy → Pygame to watch it play.

## Quick start

```bash
pip install -r requirements.txt
python play.py                 # play it yourself
python random_play.py          # random-click baseline
python train.py                # train (use Kaggle GPU for real runs)
python model_play2.py path/to/best_model.zip   # watch a trained model
```

Board size is set in `config.py` (`LEVEL = BEGINNER` 9x9/10 mines, `INTERMEDIATE` 16x16/40, `EXPERT` 16x30/99). Everything reads it, so change it in one place.

## Files

| File | What it does |
|------|--------------|
| `engine.py` | The game. `MineSweeper.reset()`, `open(x, y)`, `flag(x, y)`. `board` is what the player sees. |
| `minesweeper_env.py` | `MineEnv`: wraps the engine as a Gymnasium env (obs, actions, rewards, masks). |
| `minePolicy.py` | Current policy: fully convolutional (`ConvTrunk` + `ConvHeads` → `MinePolicy`). |
| `featureCNN.py` | Older policy: CNN then a Linear layer (`FeatureCNN`). Used by `model_play.py`. |
| `train.py` | MaskablePPO training with eval, checkpoint and milestone callbacks. |
| `milestone_callback.py` | Saves the model at chosen PPO iterations (0, 1, 10, 50…), handy for "before vs after" video clips. |
| `model_play.py` / `model_play2.py` | Watch a saved model play. `model_play` = old FeatureCNN model, `model_play2` = full-conv model. |
| `play.py`, `random_play.py` | Human play, random baseline. |
| `train_kaggle.ipynb` | Kaggle version of training. It has its **own copies** of the engine/env/policy code in cells, so edits to the `.py` files must be copied there by hand. |
| `test.py` | Scratch file for checking tensor shapes. |

## How it works

### Board encoding (`engine.py`)
`board` cell values: `-2` hidden, `-1` mine/flag, `0`–`8` revealed number.

### Observation
One-hot of the board: shape `(11, rows, cols)`, one channel per value from `-2` to `8`. The network sees "is this cell hidden?", "is it a 3?", etc. as separate image channels.

### Actions
`Discrete(3 * rows * cols)`. Action index → `kind, cell = divmod(action, rows*cols)`, then `row, col = divmod(cell, cols)`.
- kind 0 = open the cell. Kinds 1 and 2 exist in the action space but are **masked off** in `action_masks()`, so the agent can only open cells right now.
- The mask allows only hidden cells, so the agent never wastes a move clicking an open cell.

### Rewards (`MineEnv.step`)
| Event | Reward |
|-------|--------|
| Hit a mine | -1.0, game over |
| Win | +1.0, game over |
| Move opened at least one cell | +0.12 |
| Move opened nothing (not game over) | -0.1 |
| "Guess": clicked a cell with no revealed neighbours (not the first move) | -0.3 extra |

Episodes are cut off (`truncated`) after `rows*cols` steps. `info["is_success"]` reports the win, which SB3 logs as the success rate.

### Policy (`minePolicy.py`)
- `ConvTrunk`: 4 conv layers (3x3, 64 channels, padding 1), so the output stays board-sized: 64 features per cell. Each cell's features come from its neighbourhood, which is how Minesweeper logic works.
- `ConvHeads` actor: a 1x1 conv turns each cell's 64 features into 3 logits (one per action kind) → flattened into the action logits. No Linear layer, so the same weights work for every cell.
- `ConvHeads` critic: averages features over the whole board → Linear(64) → value.
- `MinePolicy` plugs these into SB3's `MaskableActorCriticPolicy`.

### Training (`train.py`)
- 4 parallel envs (`SubprocVecEnv`), 20M timesteps, `batch_size=1024`, `gamma=0.95`, `ent_coef=0.01`.
- Output goes to `runs/<RUN>/` locally or `/kaggle/working/<RUN>/` on Kaggle:
  - `best/best_model.zip`: best eval score (eval every 50k steps, 200 games)
  - `checkpoints/`: every 500k steps
  - `milestones/iter_N.zip`: at chosen PPO iterations
  - `final_model.zip`
- Trained results are kept in `results/` (git-ignored).

## Notes
- The VPS has no GPU: only short CPU smoke tests there. Real training runs on Kaggle.
- Changing `LEVEL` changes the obs/action shapes, so old models won't load on a new board size.
