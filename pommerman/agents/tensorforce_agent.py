"""
A Work-In-Progress agent using Tensorforce
"""
from . import BaseAgent
from .. import characters


class TensorForceAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""

    def __init__(self, character=characters.Bomber, algorithm='ppo'):
        super(TensorForceAgent, self).__init__(character)
        self.algorithm = algorithm

    def act(self, obs, action_space):
        """This agent has its own way of inducing actions. See train_with_tensorforce."""
        return None

    def initialize(self, env, obs_shape=None):
        from gym import spaces
        from tensorforce.agents import PPOAgent
        from tensorforce.agents import DQFDAgent

        if self.algorithm == "ppo":
            if type(env.action_space) == spaces.Tuple:
                actions = {
                    str(num): {
                        'type': int,
                        'num_actions': space.n
                    }
                    for num, space in enumerate(env.action_space.spaces)
                }
            else:
                actions = dict(type='int', num_actions=env.action_space.n)

            if obs_shape:
                shape = obs_shape
            else:
                shape = env.observation_space.shape
            return PPOAgent(
                states=dict(type='float', shape=shape),
                actions=actions,
                network=[
                    dict(type='dense', size=256),
                    dict(type='dense', size=128),
                    dict(type='dense', size=64)
                ],
                batching_capacity=1000,
                discount=0.99,
                step_optimizer=dict(type='adam', learning_rate=1e-4))


        if self.algorithm == "dqfd":
            if type(env.action_space) == spaces.Tuple:
                actions = {
                    str(num): {
                        'type': int,
                        'num_actions': space.n
                    }
                    for num, space in enumerate(env.action_space.spaces)
                }
            else:
                actions = dict(type='int', num_actions=env.action_space.n)

            if obs_shape:
                shape = obs_shape
            else:
                shape = env.observation_space.shape
            return DQFDAgent(
                states=dict(type='float', shape=shape),
                actions=actions,
                network=[
                    dict(type='dense', size=256),
                    dict(type='dense', size=128),
                    dict(type='dense', size=64)
                ],
                batching_capacity=1000,
                discount=0.99,
                optimizer=dict(type='adam', learning_rate=1e-4),
                memory=dict(type='replay', capacity=10000, include_next_states=True),
                actions_exploration=dict(type="epsilon_decay",initial_epsilon= 0.5,final_epsilon=0.0,timesteps=10000)
                )
        return None
