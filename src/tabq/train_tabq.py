import sys
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# train_tabq.py
from pathlib import Path
import numpy as np
import random

from env_factory import create_environment
from agent_tabq import CustomWrapper
from config import Config
from tabq import QTable
from featurizer import encode_state
from controller import ShooterController


def epsilon_greedy(q_values: np.ndarray, mask: np.ndarray, epsilon: float) -> int | None:
    """ε-greedy over masked actions """
    valid = np.where(mask == 1)[0]
    if valid.size == 0:
        return None
    if random.random() < epsilon:
        return int(random.choice(valid))
    masked = np.full(4, -np.inf, dtype=float)
    masked[valid] = q_values[valid]
    return int(np.argmax(masked))


def train(
    output_dir: str = "results/tabq",
    num_episodes: int = 200,
    max_steps: int = 2000,
    A: int = 6,
    B: int = 3,
    gamma: float = 0.99,
    alpha: float = 0.5,
    epsilon: float = 1.0,
):
    # build env (single archer agent)
    env = create_environment(num_agents=1, visual_observation=False)
    env = CustomWrapper(env)

    controller = ShooterController(env)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config(A=A, B=B, gamma=gamma, qtable_path=str(out / "qtable.npz"))
    cfg.save(out / "config.json")

    C = A * B + 1
    qtab = QTable(C=C, num_actions=4)

    total_steps = 0

    for ep in range(num_episodes):
        env.reset(seed=None)

        prev_state = None
        prev_action = None

        step_in_ep = 0
        ep_return = 0.0
        for agent in env.agent_iter(max_iter=max_steps):
            obs, reward, terminated, truncated, info = env.last()

            ep_return += reward

            # If we had a pending (s, a), update it now with (reward, obs as s')
            if prev_state is not None and prev_action is not None:
                # featurize next state from the current obs
                s_prime, mask_prime, _ = encode_state(obs, A=A, B=B, k=4)
                valid_next = np.where(mask_prime == 1)[0]
                if valid_next.size > 0:
                    q_next = qtab.q(s_prime)
                    max_next = float(np.max(q_next[valid_next]))
                else:
                    max_next = 0.0
                target = float(reward) + gamma * max_next
                qtab.update(prev_state, prev_action, target, alpha)
                prev_state, prev_action = None, None  # consumed

            if terminated or truncated:
                # Must pass None for deadent
                env.step(None)
                continue

            # Choose a slot action from current obs
            s, mask, _ = encode_state(obs, A=A, B=B, k=4)
            q_vals = qtab.q(s)
            a = epsilon_greedy(q_vals, mask, epsilon)

            if a is None:
                env_action = 5  #niks
            else:
                controller.set_target_slot(a)
                env_action = controller(observation=obs, agent=agent)
            #  time we see this agent
            if a is not None:
                prev_state, prev_action = s, a

            env.step(env_action)
            step_in_ep += 1
            total_steps += 1

        # decays
        epsilon = max(0.05, epsilon * 0.995)
        alpha = max(0.1, alpha * 0.999)

        print(
            f"Episode {ep + 1}/{num_episodes} | steps={step_in_ep} "
            f"| return={ep_return:.2f} | ε={epsilon:.3f} α={alpha:.3f}"
        )

    # save artifacts
    qtab.save(out / "qtable.npz")
    np.savez(out / "checkpoint.npz", steps=np.array([total_steps], dtype=np.int64))


if __name__ == "__main__":
    train()
