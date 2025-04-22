import numpy as np
import torch

import argparse

from agent import DQNAgent
from catmouseenv import CatMouseEnv
from config import INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM, EPSILON_START

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------


def evaluate(model_path: str, episodes: int = 10, render: bool = False):
    env = CatMouseEnv()
    agent = DQNAgent(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM, epsilon=EPSILON_START)
    agent.load(model_path)
    agent.eval_mode()

    rewards = []

    for ep in range(episodes):
        state_np = env.reset()
        state = torch.tensor(state_np, dtype=torch.float32)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            if render:
                env.render()

            neighbors = env.available_mouse_actions()
            action_node = agent.act(state, neighbors, eval=True)
            next_state_np, reward, done = env.step(action_node)
            state = torch.tensor(next_state_np, dtype=torch.float32)
            total_reward += reward
            steps += 1

        print(f"Episode {ep + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
        rewards.append(total_reward)

    print(f"\nAverage reward over {episodes} episodes: {np.mean(rewards):.3f}")


def evaluate_wins(model_path: str, episodes: int = 1000, render: bool = False):
    env = CatMouseEnv()
    agent = DQNAgent(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM, epsilon=EPSILON_START)
    agent.load(model_path)
    agent.eval_mode()

    rewards = []
    wins = 0
    losses = 0
    neutrals = 0

    for ep in range(episodes):
        state_np = env.reset()
        state = torch.tensor(state_np, dtype=torch.float32)
        done = False
        total_reward = 0

        while not done:
            if render:
                env.render()

            neighbors = env.available_mouse_actions()
            action_node = agent.act(state, neighbors, eval=True)
            next_state_np, reward, done = env.step(action_node)
            state = torch.tensor(next_state_np, dtype=torch.float32)
            total_reward += reward

        rewards.append(total_reward)

        # Win/loss logic — adjust this based on your actual reward system
        if total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        else:
            neutrals += 1

    print(f"\nEvaluation Summary over {episodes} episodes:")
    print(f"  Wins   : {wins} ({wins / episodes:.2%})")
    print(f"  Losses : {losses} ({losses / episodes:.2%})")
    print(f"  Neutral: {neutrals} ({neutrals / episodes:.2%})")
    print(f"  Avg Reward: {np.mean(rewards):.3f}")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    evaluate_wins(args.model_path, args.episodes, args.render)
