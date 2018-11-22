"""
A Work-In-Progress agent using Tensorforce
"""
from . import BaseAgent
from .. import characters


class TensorForceLoadedAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""

    def __init__(self, character=characters.Bomber, algorithm='ppo'):
        super(TensorForceLoadedAgent, self).__init__(character)
        self.algorithm = algorithm
        self.ppo_agent = None
        self.gym = None

    def act(self, obs, action_space):
        """This agent has its own way of inducing actions. See train_with_tensorforce."""
        return self.ppo_agent.act(self.gym.featurize(obs),
                                  deterministic=True, independent=True)

    def initialize(self, env, path):
        from gym import spaces
        from tensorforce.agents import PPOAgent

        self.gym = env

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

            self.ppo_agent = PPOAgent(
                states=dict(type='float', shape=env.observation_space.shape),
                actions=actions,
                network=[
                    dict(type='dense', size=64),
                    dict(type='dense', size=128),
                    dict(type='dense', size=64)
                ],
                batching_capacity=1000,
                discount=0.99,
                step_optimizer=dict(type='adam', learning_rate=1e-4))

            self.ppo_agent.restore_model(directory=path)
            print("Model loaded from {}".format(path))

            return self.ppo_agent
        return None
