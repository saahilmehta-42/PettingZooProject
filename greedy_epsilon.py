import glob
import os
import time
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import pettingzoo.utils
from pettingzoo.classic import connect_four_v3


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        self.observation_space = super().observation_space(self.possible_agents[0])["observation"]
        self.action_space = super().action_space(self.possible_agents[0])
        return self.observe(self.agent_selection), {}

    def step(self, action):
        super().step(action)
        obs, reward, termination, truncation, info = super().last()
        return obs, -reward, termination, truncation, info

    def observe(self, agent):
        return super().observe(agent)["observation"]

    def action_mask(self):
        return super().observe(self.agent_selection)["action_mask"]

def mask_fn(env):
    return env.action_mask()

def train_action_mask(env_fn, steps=10_000, seed=0, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, **env_kwargs):
    env = env_fn.env(**env_kwargs)
    print(f"Starting training on {str(env.metadata['name'])}.")
    env = SB3ActionMaskWrapper(env)
    env.reset(seed=seed)
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model.set_random_seed(seed)
    epsilon = epsilon_start

    for step in range(steps):
        current_agent = env.agent_selection
        obs = env.observe(current_agent)
        action_mask = mask_fn(env)

        if np.random.rand() < epsilon:
            valid_actions = np.where(action_mask)[0]
            action = np.random.choice(valid_actions)
        else:
            action, _states = model.predict(obs, action_masks=action_mask, deterministic=True)

        next_obs, reward, termination, truncation, info = env.step(action)

        model.learn(total_timesteps=1)

        if termination or truncation:
            env.reset(seed=seed)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")
    env.close()


def eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs):
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(f"Starting evaluation. Trained agent (Player 0) will play as {env.possible_agents[0]}.")

    try:
        latest_policy = max(glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime)
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = MaskablePPO.load(latest_policy)

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    def sample_action(action_space, action_mask):
        valid_actions = np.where(action_mask)[0]
        return np.random.choice(valid_actions)

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            observation, action_mask = obs.values()

            if termination or truncation:
                if (env.rewards[env.possible_agents[0]] != env.rewards[env.possible_agents[1]]):
                    winner = max(env.rewards, key=env.rewards.get)
                    scores[winner] += env.rewards[winner]
                for a in env.possible_agents:
                    total_rewards[a] += env.rewards[a]
                round_rewards.append(env.rewards)
                break
            else:
                if agent == env.possible_agents[0]:
                    act = int(model.predict(observation, action_masks=action_mask, deterministic=True)[0])
                else:
                    act = sample_action(env.action_space(agent), action_mask)
            env.step(act)
    env.close()

    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[0]] / sum(scores.values())
    print("Rewards by round: ", round_rewards)
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return round_rewards, total_rewards, winrate, scores


def human_vs_agent(env_fn, render_mode="human", **env_kwargs):
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(f"Starting a game against the trained agent.")

    try:
        latest_policy = max(glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime)
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = MaskablePPO.load(latest_policy)

    env.reset()

    def sample_action(action_space, action_mask):
        valid_actions = np.where(action_mask)[0]
        return np.random.choice(valid_actions)

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        observation, action_mask = obs.values()

        if termination or truncation:
            break
        else:
            if agent == env.possible_agents[0]:
                act = int(model.predict(observation, action_masks=action_mask, deterministic=True)[0])
            else:
                print("Your turn! Here's the current state of the board:")
                print(observation)
                while True:
                    try:
                        act = int(input(f"Select an action (0-{len(action_mask)-1}): "))
                        if action_mask[act] == 1:
                            break
                        else:
                            print("Invalid action. Try again.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
            env.step(act)
    env.close()
    print("Game over.")


if __name__ == "__main__":
    env_fn = connect_four_v3

    env_kwargs = {}

    # Evaluation/training hyperparameter notes:
    # 10k steps: Winrate:  0.76, loss order of 1e-03
    # 20k steps: Winrate:  0.86, loss order of 1e-03
    
    train_action_mask(env_fn, steps=100, seed=34, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.5, **env_kwargs)
    
    eval_action_mask(env_fn, num_games=100, **env_kwargs)
    
    human_vs_agent(env_fn, **env_kwargs)