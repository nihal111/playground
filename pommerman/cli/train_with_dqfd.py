"""Train an agent with TensorForce.

Call this with a config, a game, and a list of agents, one of which should be a
tensorforce agent. The script will start separate threads to operate the agents
and then report back the result.

An example with all three simple agents running ffa:
python3.6 train_with_dqfd.py --agents=tensorforce::dqfd,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent --config=PommeFFACompetition-v0
"""
import atexit
import functools
import os
import pickle as pk
import glob
import argparse
import docker
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

from pommerman import helpers, make
from pommerman.agents import TensorForceAgent

import time


CLIENT = docker.from_env()
wins = 0
losses = 0
avg_timesteps = 0.0
MODEL_SAVE_PATH = "./saved-dqfd/"


def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]


class WrappedEnv(OpenAIGym):
    '''An Env Wrapper used to make it easier to work
    with multiple agents'''

    def __init__(self, gym, agent_id, visualize=False):
        self.gym = gym
        self.visualize = visualize
        self.agent_id = agent_id

    def execute(self, action):
        if self.visualize:
            self.gym.render()

        actions = self.unflatten_action(action=action)

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = self.gym.featurize_special(
            state[self.gym.training_agent])

        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = self.gym.featurize_special(obs[self.agent_id])
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

    env = make(config, agents, game_state_file)
    training_agent = None

    for agent in agents:
        if type(agent) == TensorForceAgent:
            training_agent = agent
            env.set_training_agent(agent.agent_id)
            break

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    # Create a Proximal Policy Optimization agent
    agent = training_agent.initialize(env, obs_shape=(203,))

    # Callback function printing episode statistics
    def episode_finished(r):
        if r.episode % 100 == 0:
            agent.save_model(directory=MODEL_SAVE_PATH)
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
    wrapped_env = WrappedEnv(env, training_agent.agent_id,
                             visualize=args.render)

    if (os.path.exists(MODEL_SAVE_PATH) and
            len(os.listdir(MODEL_SAVE_PATH)) > 0):
        agent.restore_model(directory=MODEL_SAVE_PATH)
    else:
        start_time = time.time()
        print("Loading recorded experiences")
        files = glob.glob('demonstrationsDQFD/*.pkl')
        demonstrations = list()
        for i in range(len(files)):
            file = files[i]
            with open(file, 'rb') as f:
                x = pk.load(f)
                for timestep in range(len(x)):
                    demonstration = dict(
                        states=x[timestep][0],
                        internals=agent.current_internals,
                        actions=x[timestep][1],
                        terminal=x[timestep][3],
                        reward=x[timestep][2])
                    demonstrations.append(demonstration)

            if (i > 0 and i % 20 == 0):
                now = time.time()
                elapsed = time.strftime("%H:%M:%S",
                                        time.gmtime(now - start_time))
                print("Starting Pre-training {} at {}".
                      format(i, elapsed))
                agent.import_demonstrations(demonstrations=demonstrations)
                agent.pretrain(steps=10000)
                agent.save_model(directory=MODEL_SAVE_PATH)
                demonstrations = list()
                print("Finished Pre-training")
                runner = Runner(agent=agent, environment=wrapped_env)
                runner.run(
                    episodes=1, max_episode_timesteps=2000, testing=True)

    runner = Runner(agent=agent, environment=wrapped_env)
    runner.run(episodes=5000, max_episode_timesteps=2000,
               episode_finished=episode_finished)
    print("Losses: {}, Wins: {}, Avg timesteps: {}".format(
        losses, wins, avg_timesteps))

    agent.save_model(directory=MODEL_SAVE_PATH)

    # print("Stats: ", runner.episode_rewards, runner.episode_timesteps,
    #       runner.episode_times)

    agent.restore_model(directory=MODEL_SAVE_PATH)
    runner = Runner(agent=agent, environment=wrapped_env)
    runner.run(episodes=1, max_episode_timesteps=2000, testing=True)

    # print("Stats: ", runner.episode_rewards, runner.episode_timesteps,
    #       runner.episode_times)

    try:
        runner.close()
    except AttributeError as e:
        pass


if __name__ == "__main__":
    main()
