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
    
    mehta = 0
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent."""
        super().reset(seed, options)

        # Strip the action mask out from the observation space
        self.observation_space = super().observation_space(self.possible_agents[0])["observation"]
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info."""
        super().step(action)
        obs, reward, termination, truncation, info = super().last()

        # Return only observation, reward, termination, truncation, info
        return obs, -reward, termination, truncation, info
    
    
    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().observe(self.agent_selection)["action_mask"]


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask()


def train_action_mask(env_fn, steps=10_000, seed=0, epsilon=0.1, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    env = env_fn.env(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = SB3ActionMaskWrapper(env)

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
    
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model.set_random_seed(seed)
        
    # Training loop with epsilon-greedy exploration
    for _ in range(steps):
        obs = env.observe(env.agent_selection)
        action_mask = env.action_mask()

        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(np.where(action_mask == 1)[0])
        else:
            action = int(model.predict(obs, action_masks=action_mask, deterministic=True)[0])

        next_obs, reward, termination, truncation, info = env.step(action)
        
        model.learn(total_timesteps=1)

        # Handle termination or truncation
        if termination or truncation:
            env.reset()

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()


def eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs):
    # Evaluate a trained agent (Player 0) vs a random agent (Player 1)
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(
        f"Starting evaluation. Trained agent (Player 0) will play as {env.possible_agents[0]}."
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = MaskablePPO.load(latest_policy)

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Separate observation and action mask
            observation, action_mask = obs.values()

            if termination or truncation:
                # If there is a winner, keep track, otherwise don't change the scores (tie)
                if (
                    env.rewards[env.possible_agents[0]]
                    != env.rewards[env.possible_agents[1]]
                ):
                    winner = max(env.rewards, key=env.rewards.get)
                    scores[winner] += env.rewards[
                        winner
                    ]  # only tracks the largest reward (winner of game)
                # Also track negative and positive rewards (penalizes illegal moves)
                for a in env.possible_agents:
                    total_rewards[a] += env.rewards[a]
                # List of rewards by round, for reference
                round_rewards.append(env.rewards)
                break
            else:
                if agent == env.possible_agents[0]:
                    # Trained model predicts action for Player 0
                    act = int(
                        model.predict(
                            observation, action_masks=action_mask, deterministic=True
                        )[0]
                    )
                else:
                    # Player 1 (random agent) selects actions randomly
                    act = env.action_space(agent).sample(action_mask)
            env.step(act)
    env.close()

    # Avoid dividing by zero
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
    # Allow a human to play against the trained agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(f"Starting a game against the trained agent.")

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = MaskablePPO.load(latest_policy)

    env.reset()

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        # Separate observation and action mask
        observation, action_mask = obs.values()

        if termination or truncation:
            break
        else:
            if agent == env.possible_agents[0]:
                # Trained model predicts action for Player 0
                act = int(
                    model.predict(
                        observation, action_masks=action_mask, deterministic=True
                    )[0]
                )
            else:
                # Human player selects action
                print("Your turn! Here's the current state of the board:")
                print(observation)  # Display the observation for the human player
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

    # Train a model against itself with epsilon-greedy exploration
    train_action_mask(env_fn, steps=100, seed=23, epsilon=0.1, **env_kwargs)

    # Evaluate the trained agent against a random agent
    eval_action_mask(env_fn, num_games=1000, render_mode=None, **env_kwargs)

    # Play a game against the trained agent
    human_vs_agent(env_fn, render_mode="human", **env_kwargs)
    
    