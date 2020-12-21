import tensorflow as tf
import gym
import numpy as np

from config import Config as cfg
from utils import vpg_loss, rewards_to_go, mlp

env = gym.make('CartPole-v0')
obs_dim = env.observation_space.shape[0]
n_acts = env.action_space.n
logits_network = mlp(obs_dim, sizes=[32, n_acts], activation=tf.nn.tanh)
baseline_network = mlp(obs_dim, sizes=[32, 1], activation=tf.nn.tanh)

acts = tf.random.categorical(logits=logits_network.output, num_samples=1)
policy_network = tf.keras.Model(inputs=logits_network.inputs, outputs=[logits_network.outputs[0], acts])

optimizer = tf.optimizers.Adam(learning_rate=cfg.learning_rate)
baseline_optimizer = tf.optimizers.Adam(learning_rate=cfg.learning_rate)
total_loss = []

def train_step(policy_network, optimizer, epsilon, baseline_network = None, baseline_optimizer = None):
    loss = 0

    batch_obs = []  # for observations
    batch_acts = []  # for actions
    batch_logits = []  # store logits
    batch_weights = []  # for R(tau) weighting in policy gradient
    batch_rets = []  # for measuring episode returns
    batch_lens = []  # for measuring episode lengths

    # reset episode-specific variables
    obs = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over
    ep_rews = []  # list for rewards accrued throughout ep

    while True:
        batch_obs.append(obs.copy())
        logit, act = policy_network(obs.reshape(1,-1))

        if np.random.binomial(1, epsilon):
            act = env.action_space.sample()
        else:
            act = act.numpy()[0][0]

        obs, rew, done, _ = env.step(act)

        batch_acts.append(act)
        batch_logits.append(logit)
        ep_rews.append(rew)

        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a|s) is R(tau)
            # batch_weights += [ep_ret] * ep_len
            batch_weights += [np.sum([r*cfg.discount_rate**j for j, r in zip(range(len(ep_rews[i:])), ep_rews[i:])])
                              for i in range(len(ep_rews))]

            # reset episode-specific variables
            obs, done, ep_rews = env.reset(), False, []

            # end experience loop if we have enough of it
            if len(batch_obs) > cfg.batch_size:
                # print(len(batch_obs))
                break

    if baseline_network:
        with tf.GradientTape() as tape:
            baselines = tf.squeeze(baseline_network(np.array(batch_obs)))
            baseline_loss_value = tf.losses.mean_squared_error(baselines, batch_weights)

        baseline_gradients = tape.gradient(baseline_loss_value, baseline_network.trainable_variables)
        baseline_optimizer.apply_gradients(zip(baseline_gradients, baseline_network.trainable_variables))
    else:
        baselines = None

    with tf.GradientTape() as tape:
        probs, _ = policy_network(np.array(batch_obs))
        loss_value = vpg_loss(batch_weights, batch_acts, probs, n_acts, baselines)

    gradients = tape.gradient(loss_value, policy_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables))

    return({'loss_value':loss_value,
            'probs':probs,
            'batch_obs':batch_obs,
            'batch_acts':batch_acts,
            'batch_logits':batch_logits,
            'batch_weights':batch_weights,
            'batch_rets':batch_rets,
            'batch_lens':batch_lens
            })

for i in range(cfg.epochs):
    res = train_step(policy_network, optimizer,
                     epsilon=0,
                     #epsilon=max(cfg.explore_0*cfg.explore_decay**i, 0.05),
                     baseline_network=baseline_network,
                     baseline_optimizer=baseline_optimizer)
    total_loss.append(res['loss_value'])
    print('Epoch {} | Current loss:  {}, Average Loss:  {}, Average Reward:  {}'.format(
        i, res['loss_value'], np.sum(total_loss)/len(total_loss), np.sum(res['batch_rets'])/len(res['batch_rets'])))

import time

obs = env.reset()  # first obs comes from starting distribution
done = False  # signal from environment that episode is over

for i in range(10):
    while not done:
        logit, act = policy_network(obs.reshape(1, -1))
        obs, rew, done, _ = env.step(act.numpy()[0][0])
        env.render()
        time.sleep(0.01)

    done = False
    obs = env.reset()

env.close()