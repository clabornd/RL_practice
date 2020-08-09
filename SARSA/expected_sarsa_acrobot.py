from config import Config as cfg
from utils import *

import tensorflow as tf
import gym
import numpy as np
import datetime
import itertools

# tensorflow nonsense
physical_devices = tf.config.experimental.list_physical_devices(device_type = "GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#
env = gym.make('Acrobot-v1')

num_tilings = 48

tilings = [[] for _ in range(num_tilings)]
bins = [6]*4
sub_obs = [0,2,4,5]

steps = []
for i in sub_obs:
    high = env.observation_space.high[i]
    low = env.observation_space.low[i]
    steps.append((high - low)/(num_tilings/4-1))

for i in reversed(range(4)):
    sub_tilings = itertools.combinations(sub_obs, i+1)
    for st in sub_tilings:
        for k, j in enumerate(st):
            tmp_tiling = np.linspace(env.observation_space.low[j] - steps[k] * i / (num_tilings - 1), \
                                     env.observation_space.high[j] + steps[k] * (1 - i / (num_tilings - 1)),
                                    bins[k])
            tilings[i].append(tmp_tiling)

for t in tilings:

for i, t in enumerate(tilings[0]):
    tmp_obs = obs[sub_obs]
    if(i == 0):
        out_arr = discretize(tmp_obs[i], t)
    else:
        out_arr = np.outer(out_arr, discretize(tmp_obs[i], t)).reshape(-1)




n_acts = env.action_space.n
obs_dim = env.observation_space.shape[0]

# define network, just a linear combination of the flattened tilings
Q1 = mlp(obs_dim, [144, n_acts], dropout=None, use_bias=True, activation = tf.nn.sigmoid)

# ...and optimizer
Q1_opt = tf.optimizers.Adam(learning_rate=cfg.learning_rate)

# set up tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time + '/train'
tb = tf.summary.create_file_writer(train_log_dir)

# returns the flattened tiling representation of an observation
def get_tiling_input(ptilings, vtilings, obs):
    pos_input = []
    for tiling in ptilings:
        pos_input.append(discretize(obs[0], tiling))

    speed_input = []
    for tiling in vtilings:
        speed_input.append(discretize(obs[1], tiling))

    input = np.concatenate([np.outer(p, v).reshape(-1) for p, v in zip(pos_input, speed_input)])

    return input

#
def train_step(Q1, Q1_opt, epsilon, epoch):
    '''
    :param Q1:  First function approximator for the action value function
    :param Q1_opt: Optimizer for the first function approximator
    :param epsilon: Agent will explore with probability epsilon
    :param epoch: the epoch, for logging purposes
    :return: dictionary of summary statistics of the train step
    '''

    # Initialization for batch.  Batch contains multiple episodes:
    collect_Q1_loss = 0 # loss for Q1 and Q2
    iteration = 0 # for logging
    batch_acts = []  # store actions
    batch_rews = []  # store rewards
    batch_obs = [] # store observations
    batch_rets = []  # store episode returns
    batch_lens = []  # store episode lengths
    batch_speed = [] # store the absolute speed over the train step

    # reset episode-specific variables
    obs = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over
    ep_rews = []  # list for rewards accrued throughout ep
    ep_obs = [] # every flattened state representation
    ep_acts = [] #
    ep_speed = [] #

    while not done:
        # the time step (t) in the book
        step = 0 # this resets on episode end, iteration doesn't

        # get initial input
        input = np.array(obs)
        ep_obs.append(input.copy().reshape(1,-1))

        tau = -1 # the time step being updated, intialize to nonsense value

        # initial action
        if np.random.binomial(1, 1 - epsilon):
            act = np.argmax(Q1(input.reshape(1,-1)))
        else:
            act = env.action_space.sample()
        ep_acts.append(act)

        # generate an episode
        while tau != len(ep_acts) - 1:
            if done:
                ep_complete = -np.cos(env.state[0]) - np.cos(env.state[1] + env.state[0]) > 1.
                if ep_complete: # episode ended because we reached the top
                    print('top reached')
                else: # episode ended because we reached the time limit
                    # this is so that there is an observation at tau + td_steps that we can compute the Q-value of
                    input = np.array(obs)
                    ep_obs.append(input.copy().reshape(1, -1))
            else:
                obs, rew, done, _ = env.step(act)  # take action, observe new state and reward
                ep_rews.append(rew)  # store next reward
                ep_speed.append(np.abs(obs[-2])) # store speed for logging

                # store next state
                input = np.array(obs)
                ep_obs.append(input.copy().reshape(1,-1))

                # only get next action if we are not in terminal state
                if not done:
                    act = np.argmax(Q1(input.reshape(1,-1)))
                    ep_acts.append(act)

                ep_complete = False

            ## Begin update ##

            tau = (step - cfg.td_steps + 1) # the time step being updated, as described in Sutton and Barrow

            if(tau >= 0):
                # observed discounted rewards
                G = n_step_G(ep_rews[tau:], cfg.td_steps, cfg.discount_rate)[0]

                # if we have not reached end of episode, append the Q-value at time tau + td_steps ...
                if not done:
                    extra = cfg.discount_rate ** cfg.td_steps * tf.reduce_mean(Q1(ep_obs[tau + cfg.td_steps]))
                # otherwise append the last observed action value (ep_obs is end-padded with the last observation \
                # such that ep_obs[tau + cfg.td_steps] references at actual value if we are past the terminal time point
                else:
                    discount = cfg.discount_rate ** (len(ep_obs) - tau)
                    extra = tf.reduce_mean(Q1(ep_obs[tau + cfg.td_steps]))*discount if not ep_complete else 0

                target = G + extra

                # perform update
                with tf.GradientTape() as tape:
                    phi = Q1(ep_obs[tau], training=True)[:, ep_acts[tau]]

                    value_func_loss = tf.losses.MeanSquaredError()(target[np.newaxis], phi)
                    collect_Q1_loss += value_func_loss

                Q1_grads = tape.gradient(value_func_loss, Q1.trainable_variables)
                Q1_opt.apply_gradients(zip(Q1_grads, Q1.trainable_variables))

            step += 1
            iteration += 1

        # tensorboard
        with tb.as_default():
            tf.summary.histogram('weights', Q1.layers[1].weights[0], step=cfg.batch_size*epoch + iteration)
            tf.summary.histogram('grads', Q1_grads[0], step=cfg.batch_size*epoch + iteration)

        tb.flush()
        #

        # if episode is over, record info about episode
        ep_ret, ep_len = sum(ep_rews), len(ep_rews)
        batch_rets.append(ep_ret)
        batch_lens.append(ep_len)
        batch_rews.append(ep_rews)
        batch_obs.append(ep_obs)
        batch_acts.append(ep_acts)
        batch_speed.append(np.mean(ep_speed))

        # reset episode-specific variables
        obs, done, = env.reset(), False
        ep_rews, ep_obs, ep_acts, ep_speed = [], [], [], []

        # end experience loop if we have enough of it
        if iteration > cfg.batch_size:
            # print(len(batch_obs))
            break

    return({'Q1_loss': collect_Q1_loss,
            'batch_acts':batch_acts,
            'batch_rets':batch_rets,
            'batch_lens':batch_lens,
            'average_speed':np.mean(batch_speed)
            })


# training loop
for i in range(cfg.epochs):
    epsilon = epsilon=max(cfg.explore_0*cfg.explore_decay**i, cfg.min_explore)
    res = train_step(Q1, Q1_opt, epsilon=epsilon, epoch=i)
    print('Epoch {}: Q1 loss: {}, Avg Reward: {}, Avg ep len: {}, Avg speed: {}'.format(
        i, res['Q1_loss'], np.sum(res['batch_rets'])/len(res['batch_rets']),
        np.sum(res['batch_lens']) / len(res['batch_lens']), res['average_speed']))
#

### inspect the performance of the agent:
import time
obs = env.reset()  # first obs comes from starting distribution
done = False  # signal from environment that episode is over

for i in range(10):
    while not done:
        pos_input = []
        for tiling in ptilings:
            pos_input.append(discretize(obs[0], tiling))

        speed_input = []
        for tiling in vtilings:
            speed_input.append(discretize(obs[1], tiling))

        input = np.concatenate([np.outer(p, v).reshape(-1) for p, v in zip(pos_input, speed_input)])
        # value_input = np.concatenate([np.array([obs] * 3), tf.one_hot([0, 1, 2], 3)], axis=1)
        act = np.argmax(Q1(input.reshape(1,-1)))
        # act = tf.random.categorical(tf.math.softmax(Q1(obs.reshape(1, -1)) + Q2(obs.reshape(1, -1))), 1).numpy()[0][0]
        obs, rew, done, _ = env.step(act)
        env.render()
        time.sleep(0.01)

    done = False
    obs = env.reset()

env.close()