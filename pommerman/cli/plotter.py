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


def avg_from_start(x):
    cum_x = np.cumsum(x)
    avg_x = [1.0 * cum_x[i] /
             (i + 1) for i in range(len(cum_x))]
    return avg_x


def moving_avg(x, k=100):
    if k <= 0 or not isinstance(k, int) or k > len(x):
        return x
    cum_x = np.cumsum(x)
    first_k = [1.0 * cum_x[i] /
               (i + 1) for i in range(k)]
    latter = [(cum_x[i] - cum_x[i - k]) / k
              for i in range(k, len(cum_x))]
    avg_x = first_k + latter
    assert len(avg_x) == len(x)
    return avg_x


# plt.plot(episode_numbers, moving_avg(timesteps))
# plt.ylabel('Average Time Steps')

plt.plot(episode_numbers, moving_avg(rewards))
plt.ylabel('Average Reward')

plt.xlabel('Episodes')
plt.show()
# plt.savefig('standard.png')
