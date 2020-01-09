from config import Config as cfg
from utils import *

import tensorflow as tf
import gym
import numpy as np

env = gym.make('MountainCar-v0')
pos_bins = 30
speed_bins = 30
obs_dim = pos_bins+speed_bins
n_acts = env.action_space.n

Q1 = mlp(obs_dim, [n_acts], dropout=[None])
Q2 = mlp(obs_dim, [n_acts], dropout=[None])

Q1_opt = tf.optimizers.Adam(learning_rate=cfg.learning_rate)
Q2_opt = tf.optimizers.Adam(learning_rate=cfg.learning_rate)

# acts = [0,1,1,2,...]
# logits = [[l1, l2, l3], [l1, l2, l3] ...]
# acts*logits = [[l1, 0, 0], [0,l2,0], ...] ==> reduce_sum() ==> [l1, l2, ...]

def train_step(Q1, Q1_opt, Q2, Q2_opt, epsilon):
    '''
    :param Q1:  First function approximator for the action value function
    :param Q1_opt: Optimizer for the first function approximator
    :param Q2:  Second function approximator for the action value function
    :param Q2_opt: Optimizer for the second function approximator
    :param epsilon: Agent will explore with probability epsilon
    :return: dictionary of summary statistics of the train step
    '''

    # Initialization for batch.  Batch contains multiple episodes:
    collect_Q1_loss = 0 # loss for Q1 and Q2
    collect_Q2_loss = 0 #
    iteration = 0
    batch_acts = []  # store actions
    batch_rets = []  # store episode returns
    batch_lens = []  # store episode lengths
    batch_velocity = [] # store the absolute speed over the train step

    # reset episode-specific variables
    obs = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over
    ep_rews = []  # list for rewards accrued throughout ep
    ep_obs = []
    ep_acts = []
    ep_velocity = []
    which_network = np.random.binomial(1, 0.5) # switch which network we are updating

    # generate an episode
    while not done:
        pos_input = discretize(obs[0], np.linspace(env.observation_space.low[0], env.observation_space.high[0] - 0.1,
                                                   pos_bins))
        speed_input = discretize(obs[1], np.linspace(env.observation_space.low[1], env.observation_space.high[1],
                                                     speed_bins))
        input = np.concatenate([pos_input, speed_input])
        ep_obs.append(input.copy())

        if np.random.binomial(1, 1 - epsilon):
            # value_input = np.concatenate([np.array([obs] * 3), tf.one_hot([0, 1, 2], 3)], axis=1)
            act = np.argmax(Q1(input.reshape(1,-1)) + Q2(input.reshape(1,-1)))
            #act = tf.random.categorical(tf.math.softmax(Q1(obs.reshape(1, -1)) + Q2(obs.reshape(1, -1))), 1).numpy()[0][
            #    0]
        else:
            act = env.action_space.sample()

        obs, rew, done, _ = env.step(act)

        ep_acts.append(act)
        ep_rews.append(rew)
        ep_velocity.append(np.abs(obs[1]))

        if done:
            if obs[0] >= 0.5: print('Final location: {}'.format(obs[0]))
            with tf.GradientTape() as tape:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                #phi_input = np.concatenate([np.array(ep_obs), tf.one_hot(ep_acts, 3)], axis = 1)
                phi_1 = Q1(np.array(ep_obs), training=True)
                phi_2 = Q2(np.array(ep_obs), training=True)

                if which_network:
                    # targets can be static...
                    # the discounted return
                    G = n_step_G(ep_rews, cfg.td_steps, cfg.discount_rate)
                    # if we have not reached the top by end of episode, append action value estimate, otherwise append 0
                    extra = phi_2[-1][np.argmax(phi_1[-1])] if(obs[0] < 0.5) else 0

                    # if we are not done, then Q is the n-step-ahead action value function output ...
                    # ... otherwise it is the value of extra
                    Q = [cfg.discount_rate ** cfg.td_steps * phi_2[j+cfg.td_steps][np.argmax(phi_1[j+cfg.td_steps])]
                         if j+cfg.td_steps<len(phi_1)
                         else cfg.discount_rate ** (len(phi_1) - j) * extra
                         for j in range(len(phi_1))]

                    # form the full targets
                    td_targets = np.array(G) + np.array(Q)

                    # values must be linked by backprop, are they?
                    # td_values = tf.reduce_sum(phi_2*tf.one_hot(ep_acts, 3),axis=1)[:-1]
                    td_values = [phi_1[j][ep_acts[j]] for j in range(len(ep_rews))]
                    value_func_loss = tf.losses.MeanSquaredError()(td_targets, td_values)
                    collect_Q1_loss += value_func_loss

                    grads = tape.gradient(value_func_loss, Q1.trainable_variables)
                    Q1_opt.apply_gradients(zip(grads, Q1.trainable_variables))
                else:
                    G = n_step_G(ep_rews, cfg.td_steps, cfg.discount_rate)
                    extra = phi_1[-1][np.argmax(phi_2[-1])] if (obs[0] < 0.5) else 0

                    Q = [cfg.discount_rate ** cfg.td_steps * phi_1[j + cfg.td_steps][np.argmax(phi_2[j + cfg.td_steps])]
                         if j + cfg.td_steps < len(phi_2)
                         else cfg.discount_rate ** (len(phi_2) - j) * extra
                         for j in range(len(phi_2))]
                    td_targets = np.array(G) + np.array(Q)

                    td_values = [phi_2[j][ep_acts[j]] for j in range(len(ep_rews))]
                    value_func_loss = tf.losses.MeanSquaredError()(td_targets, td_values)
                    collect_Q2_loss += value_func_loss

                    grads = tape.gradient(value_func_loss, Q2.trainable_variables)
                    Q2_opt.apply_gradients(zip(grads, Q2.trainable_variables))

                iteration += len(ep_obs)
                batch_acts += ep_acts
                batch_velocity.append(np.mean(ep_velocity))

                # reset episode-specific variables
                obs, done, = env.reset(), False
                ep_rews, ep_obs, ep_acts, ep_velocity = [], [], [], []
                which_network = not which_network

                # end experience loop if we have enough of it
                if iteration > cfg.batch_size:
                    # print(len(batch_obs))
                    break

    return({'Q1_loss': collect_Q1_loss,
            'Q2_loss': collect_Q2_loss,
            'batch_acts':batch_acts,
            'batch_rets':batch_rets,
            'batch_lens':batch_lens,
            'average_velocity':np.mean(batch_velocity)
            })


for i in range(cfg.epochs):
    res = train_step(Q1, Q1_opt, Q2, Q2_opt, epsilon=max(cfg.explore_0*cfg.explore_decay**i, 0.1))
    print('Epoch {}: Q1 loss: {}, Q2 loss: {}, Avg Reward: {}, Avg ep len: {}, Avg speed: {}'.format(
        i, res['Q1_loss'], res['Q2_loss'], np.sum(res['batch_rets'])/len(res['batch_rets']),
        np.sum(res['batch_lens']) / len(res['batch_lens']), res['average_velocity']))
#

### inspect the performance of the agent:
import time
obs = env.reset()  # first obs comes from starting distribution
done = False  # signal from environment that episode is over

for i in range(10):
    while not done:
        pos_input = discretize(obs[0], np.linspace(env.observation_space.low[0], env.observation_space.high[0] - 0.1,
                                                   pos_bins))
        speed_input = discretize(obs[1], np.linspace(env.observation_space.low[1], env.observation_space.high[1],
                                                     speed_bins))
        input = np.concatenate([pos_input, speed_input])
        # value_input = np.concatenate([np.array([obs] * 3), tf.one_hot([0, 1, 2], 3)], axis=1)
        act = np.argmax(Q1(input.reshape(1,-1))+Q2(input.reshape(1,-1)))
        # act = tf.random.categorical(tf.math.softmax(Q1(obs.reshape(1, -1)) + Q2(obs.reshape(1, -1))), 1).numpy()[0][0]
        obs, rew, done, _ = env.step(act)
        env.render()
        time.sleep(0.01)

    done = False
    obs = env.reset()

env.close()