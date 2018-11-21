from matplotlib import pyplot as plt
import numpy as np

with open('log-PSRR.txt') as f:
    lines = f.readlines()


episode_numbers = []
timesteps = []
rewards = []

for line in lines:
    if 'Finished' in line:
        i1 = line.index('Finished episode ') + \
            len('Finished episode ')
        i2 = line.index('after') - 1
        epi_no = line[i1:i2]
        episode_numbers.append(int(epi_no))

        i1 = line.index('after ') + \
            len('after ')
        i2 = line.index('timesteps') - 1
        time = line[i1:i2]
        timesteps.append(int(time))

        i1 = line.index('reward: ') + \
            len('reward: ')
        i2 = line.index(')')
        rew = line[i1:i2]
        rewards.append(int(rew))

cum_timesteps = np.cumsum(timesteps)
avg_cum_timesteps = [1.0 * cum_timesteps[i] /
                     (i + 1) for i in range(len(cum_timesteps))]

cum_rewards = np.cumsum(rewards)
avg_cum_rewards = [1.0 * cum_rewards[i] /
                   (i + 1) for i in range(len(cum_rewards))]

# plt.plot(episode_numbers, avg_cum_timesteps)
# plt.ylabel('Average Time Steps')

plt.plot(episode_numbers, avg_cum_rewards)
plt.ylabel('Average Reward')

plt.xlabel('Episodes')
plt.show()
# plt.savefig('standard.png')
