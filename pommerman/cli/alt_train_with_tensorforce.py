"""Train an agent with TensorForce.

Call this with a config, a game, and a list of agents, one of which should be a
tensorforce agent. The script will start separate threads to operate the agents
and then report back the result.

An example with all three simple agents running ffa:
python train_with_tensorforce.py \
 --agents=tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent \
 --config=PommeFFACompetition-v0
"""
import atexit
import functools
import os

import argparse
import docker
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

from pommerman import helpers, make
from pommerman.agents import TensorForceAgent
from pommerman.agents import TensorForceLoadedAgent
from pommerman.agents import SimpleAgent


CLIENT = docker.from_env()
wins = 0
losses = 0
avg_timesteps = 0.0
MODEL_SAVE_PATH = './alt-saved/'

training_agent = None
testing_agent = None
env = None
wrapped_env = None


def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]


class WrappedEnv(OpenAIGym):
    '''An Env Wrapper used to make it easier to work
    with multiple agents'''

    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize

    def set_visulaize(self, boolean):
        self.visualize = boolean

    def execute(self, action):
        if self.visualize:
            self.gym.render()

        actions = self.unflatten_action(action=action)

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = self.gym.featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[3])
        return agent_obs


def main():
    '''CLI interface to bootstrap taining'''
    parser = argparse.ArgumentParser(description="Playground Flags.")
    parser.add_argument("--game", default="pommerman", help="Game to choose.")
    parser.add_argument(
        "--config",
        default="PommeFFACompetition-v0",
        help="Configuration to execute. See env_ids in "
        "configs.py for options.")
    parser.add_argument(
        "--agents",
        default="tensorforce::ppo,test::agents.SimpleAgent,"
        "test::agents.SimpleAgent,test::agents.SimpleAgent",
        help="Comma delineated list of agent types and docker "
        "locations to run the agents.")
    parser.add_argument(
        "--agent_env_vars",
        help="Comma delineated list of agent environment vars "
        "to pass to Docker. This is only for the Docker Agent."
        " An example is '0:foo=bar:baz=lar,3:foo=lam', which "
        "would send two arguments to Docker Agent 0 and one to"
        " Docker Agent 3.",
        default="")
    parser.add_argument(
        "--record_pngs_dir",
        default=None,
        help="Directory to record the PNGs of the game. "
        "Doesn't record if None.")
    parser.add_argument(
        "--record_json_dir",
        default=None,
        help="Directory to record the JSON representations of "
        "the game. Doesn't record if None.")
    parser.add_argument(
        "--render",
        default=False,
        action='store_true',
        help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        "--game_state_file",
        default=None,
        help="File from which to load game state. Defaults to "
        "None.")
    args = parser.parse_args()

    config = args.config
    record_pngs_dir = args.record_pngs_dir
    record_json_dir = args.record_json_dir
    agent_env_vars = args.agent_env_vars
    game_state_file = args.game_state_file

    # TODO: After https://github.com/MultiAgentLearning/playground/pull/40
    #       this is still missing the docker_env_dict parsing for the agents.
    agents = [
        helpers.make_agent_from_string(agent_string, agent_id + 1000)
        for agent_id, agent_string in enumerate(args.agents.split(","))
    ]

    global testing_agent
    global training_agent
    global env
    global wrapped_env

    env = make(config, agents, game_state_file)

    for i in range(len(agents)):
        # This is the agent being actively trained
        if type(agents[i]) == TensorForceAgent:
            env.set_training_agent(agents[i].agent_id)

            # Set training_agent to the PPO agent wrapped by TensorForceAgent
            training_agent = agents[i].initialize(env)
            # If a saved model exists, load it
            if (os.path.exists(MODEL_SAVE_PATH) and
                    len(os.listdir(MODEL_SAVE_PATH)) > 0):
                training_agent.restore_model(directory=MODEL_SAVE_PATH)

        # This is the stationary, pre-trained agent
        if type(agents[i]) == TensorForceLoadedAgent:
            # Check if a saved model exists
            if (os.path.exists(MODEL_SAVE_PATH) and
                    len(os.listdir(MODEL_SAVE_PATH)) > 0):
                # Set testing_agent to the PPO agent
                # wrapped by TensorForceLoadedAgent
                testing_agent = agents[i].initialize(env, MODEL_SAVE_PATH)
            # If not, then initialize a simple agent
            else:
                # Here, testing_agent is set to an int
                testing_agent = i
                agents[i] = SimpleAgent()
                agents[i].init_agent(i, env.spec._kwargs['game_type'])
    env.set_agents(agents)

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    # Callback function printing episode statistics
    def episode_finished(r):
        if r.episode > 0 and r.episode % 10 == 0:
            global testing_agent
            global training_agent
            global env
            global wrapped_env

            training_agent.save_model(directory=MODEL_SAVE_PATH)

            if testing_agent:
                if isinstance(testing_agent, int):
                    agents[testing_agent] = TensorForceLoadedAgent()
                    agents[testing_agent].init_agent(
                        testing_agent, env.spec._kwargs['game_type'])
                    testing_agent = agents[testing_agent].initialize(
                        env, MODEL_SAVE_PATH)
                else:
                    testing_agent.restore_model(directory=MODEL_SAVE_PATH)
            env.set_agents(agents)
            wrapped_env.set_visulaize(True)

        if r.episode % 10 == 1:
            wrapped_env.set_visulaize(False)

        print("Finished episode {ep} after {ts} timesteps (reward: {reward})".
              format(ep=r.episode,
                     ts=r.episode_timestep,
                     reward=r.episode_rewards[-1]))
        global losses, wins, avg_timesteps
        if r.episode_rewards[-1] == -1:
            losses += 1
        else:
            wins += 1
        avg_timesteps = sum(r.episode_timesteps) / len(r.episode_timesteps)
        return True

    atexit.register(functools.partial(clean_up_agents, agents))
    wrapped_env = WrappedEnv(env, visualize=args.render)
    runner = Runner(agent=training_agent, environment=wrapped_env)
    runner.run(episodes=5000, max_episode_timesteps=2000,
               episode_finished=episode_finished)
    print("Losses: {}, Wins: {}, Avg timesteps: {}".format(
        losses, wins, avg_timesteps))

    training_agent.save_model(directory=MODEL_SAVE_PATH)

    # print("Stats: ", runner.episode_rewards, runner.episode_timesteps,
    #       runner.episode_times)

    # training_agent.restore_model(directory=MODEL_SAVE_PATH)
    # runner = Runner(agent=training_agent, environment=wrapped_env)
    # runner.run(episodes=1, max_episode_timesteps=2000, testing=True)

    # print("Stats: ", runner.episode_rewards, runner.episode_timesteps,
    #       runner.episode_times)

    try:
        runner.close()
    except AttributeError as e:
        pass


if __name__ == "__main__":
    main()
