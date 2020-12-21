import sys, os, re, datetime, time, itertools, argparse
sys.path.append("..")

from config import Config as cfg
from utils import *
from tf_models import acrobot_mlp

import tensorflow as tf
import gym
import numpy as np

from mlflow import log_metric, log_param, log_artifacts

import pdb

# set up acrobot
env = gym.make('Acrobot-v1')
obs = env.reset()
n_acts = env.action_space.n

obs_dim = 4 # we will convert from 6 dimensional to 4 dimensional observation space using convert_to_angle()
#cos(theta1), sin(theta1), cos(theta2), sin(theta2), ang1, ang2

# number of tilings per combination
# From Sutton et al. (1996):  3 tilings for each of the 4 variables, 2 tilings for each of 6 combinations of 2 variables\
# ... 3 tilings for the 4 combinations of 3 variables, and 12 tilings for the 4 variable combination
num_tilings = [3,2,3,12]
bins = [6]*4
sub_obs = [0,2,4,5]

# manually do this since the implementation in the paper has
obs_space_high = [np.pi, np.pi, env.observation_space.high[4], env.observation_space.high[5]]
obs_space_low = [-np.pi, -np.pi, env.observation_space.low[4], env.observation_space.low[5]]

# function to convert the 6 variable observation to 4 variables
def convert_to_angle(obs):
    phi1 = np.arctan2(obs[1], obs[0])
    phi2 = np.arctan2(obs[3], obs[2])
    return [phi1, phi2, obs[4], obs[5]]

### get the step size for each tiling ###
# the equation is (high-low)/num_bins/num_tilings for each tiling
steps = []
for nt in num_tilings:
    tmp_steps = []
    for i in sub_obs:
        if i in [4,5]:
            high = env.observation_space.high[i]
            low = env.observation_space.low[i]
        else:
            high = np.pi
            low = -np.pi
        tmp_steps.append((high - low)/6/nt)
    steps.append(tmp_steps)

### create the tilings as described in Sutton et al. (1996)
tilings = [[] for _ in range(4)]
# i + 1 here will be the number of variables used
for i, stp, nt in zip(range(len(steps)), steps, num_tilings):
    # this different combinations of variables used
    sub_tilings = itertools.combinations(range(4), i + 1)

    # for each combination
    for st in sub_tilings:
        # number of tilings per combination
        for l in range(nt):
            # for each index in the state space
            for k, j in enumerate(st):
                # tilings shifted by the step parameter
                tmp_tiling = np.linspace(obs_space_low[j] - stp[k] * l, \
                                         obs_space_high[j] + stp[k] * (nt-l-1),
                                         bins[k])
                tilings[i].append({"variable_set":st,"variable":j, "tiling_space": tmp_tiling})

'''
Create the correct input for our network.

Input is a set of tilings for each set of variables and an observation converted to [angle1, angle2, angv1, angv2]

output is length 4 list element.  [1-tuple-tilings, 2-tuple-tilings, 3-tuple-tilings, 4-tuple-tilings]

'''
def make_input(tilings, obs):
    input_vector = []

    for nvars in range(4):
        out_arrs = []
        for i, t in enumerate(tilings[nvars]):
            if(i % (nvars+1) == 0):
                tmp_arr = discretize(obs[t['variable']], t['tiling_space'])
            else:
                tmp_arr = np.outer(tmp_arr, discretize(obs[t['variable']], t['tiling_space'])).reshape(-1)

            if((i + 1) % (nvars+1) == 0):
                out_arrs.append(tmp_arr)

        input_vector.append(np.concatenate(out_arrs).reshape(1, -1))

    return(input_vector)

# define network, the dimensions are from from Sutton et al. (1996)
# Q1 = mlp(18648, [n_acts], dropout=None, use_bias=False, activation=None) # old mlp
Q1 = acrobot_mlp(sizes=[12*6, 12*6**2], n_acts=3, activations=[tf.nn.sigmoid, tf.nn.sigmoid], \
                 use_bias=False)

# ...and optimizer
Q1_opt = tf.optimizers.SGD(learning_rate=cfg.learning_rate)

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
    collect_Q1_loss = 0 # loss for Q1
    iteration = 0 # for logging
    batch_acts = []  # store actions
    batch_rews = []  # store rewards
    batch_obs = [] # store observations
    batch_rets = []  # store episode returns
    batch_lens = []  # store episode lengths
    batch_speed = [] # store the absolute speed over the train step
    batch_completions = 0 # how many times did we reach the top?

    # reset episode-specific variables
    obs = env.reset() # first obs comes from starting distribution
    done = False  # signal from environment that episode is over
    ep_rews = []  # list for rewards accrued throughout ep
    ep_obs = [] # every flattened state representation
    ep_acts = [] #
    ep_speed = [] #

    pdb.set_trace()

    while not done:
        # the time step (t) in the book
        step = 0 # this resets on episode end, iteration doesn't

        # get initial input
        input = make_input(tilings, convert_to_angle(obs))
        ep_obs.append(input.copy())

        tau = -1 # the time step being updated, intialize to nonsense value

        # initial action
        if np.random.binomial(1, 1 - epsilon):
            act = np.argmax(Q1(input))
        else:
            act = env.action_space.sample()
        ep_acts.append(act)

        # initialize eligibility_trace
        eligibility_trace = None

        # generate an episode
        while tau != len(ep_acts) - 1:
            if done:
                pdb.set_trace()
                ep_complete = -np.cos(env.state[0]) - np.cos(env.state[1] + env.state[0]) > 1.
                if ep_complete:
                    pdb.set_trace()
                if not ep_complete: # episode ended because we reached the time limit
                    # this is so that there is an observation at tau + td_steps that we can compute the Q-value of
                    input = make_input(tilings, convert_to_angle(obs))
                    ep_obs.append(input.copy())
            else:
                obs, rew, done, _ = env.step(act)  # take action, observe new state and reward
                ep_rews.append(rew)  # store next reward
                ep_speed.append(np.abs(obs[-2])) # store speed for logging

                # store next state
                input = make_input(tilings, convert_to_angle(obs))
                ep_obs.append(input.copy())

                # only get next action if we are not in terminal state
                if not done:
                    if np.random.binomial(1, 1 - epsilon):
                        act = np.argmax(Q1(input))
                    else:
                        act = env.action_space.sample()
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

                if np.isnan(value_func_loss):
                    pdb.set_trace()

                Q1_grads = tape.gradient(value_func_loss, Q1.trainable_variables)

                if not eligibility_trace:
                    eligibility_trace = Q1_grads
                else:
                    eligibility_trace = [cur + cfg.discount_rate * cfg.trace_decay * prev for cur, prev in
                                         zip(Q1_grads, eligibility_trace)]
                    eligibility_trace = [tf.clip_by_norm(g, 4)for g in eligibility_trace]

                Q1_opt.apply_gradients(zip(eligibility_trace, Q1.trainable_variables))

            if done:
                print('hi')
            step += 1
            iteration += 1

        # tensorboard
        # with tb.as_default():
        #     tf.summary.histogram('weights', Q1.layers[1].weights[0], step=cfg.batch_size*epoch + iteration)
        #     tf.summary.histogram('grads', Q1_grads[0], step=cfg.batch_size*epoch + iteration)
        #
        # tb.flush()
        #

        # if episode is over, record info about episode
        ep_ret, ep_len = sum(ep_rews), len(ep_rews)
        batch_rets.append(ep_ret)
        batch_lens.append(ep_len)
        batch_rews.append(ep_rews)
        batch_obs.append(ep_obs)
        batch_acts.append(ep_acts)
        batch_speed.append(np.mean(ep_speed))
        batch_completions += ep_complete

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
            'batch_completions': batch_completions,
            'average_speed':np.mean(batch_speed)
            })

if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--logdir", type=str, default=os.path.join('logs/TD_N_acrobot_', current_time))
    args = parser.parse_args()

    os.makedirs(args.logdir)
    # tb = tf.summary.create_file_writer(train_log_dir)

    # mlflow logging
    cfgparams = [el for el in cfg.__dict__.keys() if not re.search("__", el)]
    for k in cfgparams:
        log_param(k, cfg.__dict__[k])

    if(args.train):
        # training loop
        for i in range(cfg.epochs):
            epsilon = max(cfg.explore_0*cfg.explore_decay**i, cfg.min_explore) - cfg.min_explore
            res = train_step(Q1, Q1_opt, epsilon=epsilon, epoch=i)
            print('Epoch {}: Q1 loss: {}, Avg Reward: {}, Avg ep len: {}, Avg speed: {}, completions: {}'.format(
                i, res['Q1_loss'], np.sum(res['batch_rets'])/len(res['batch_rets']),
                np.sum(res['batch_lens']) / len(res['batch_lens']), res['average_speed'], res['batch_completions']))

            log_metric('batch completions', res['batch_completions'], step=i)
            log_metric('average ep len', res['average_speed'], step=i)

        artifacts_dir = os.path.join(args.logdir, "artifacts")
        os.makedirs(artifacts_dir)
        Q1.save_weights(os.path.join(artifacts_dir, "final_model.ckpt"))
        log_artifacts(artifacts_dir)

    ### inspect the performance of the agent:
    if(args.test):
        obs = env.reset()
        done = False

        for i in range(5):
            while not done:
                # create input, take action, step, render
                input = make_input(tilings, convert_to_angle(obs))
                act = np.argmax(Q1(input))
                obs, rew, done, _ = env.step(act)
                env.render()
                time.sleep(0.01)

            done = False
            obs = env.reset()

        env.close()
