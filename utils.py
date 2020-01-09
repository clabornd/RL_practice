import tensorflow as tf
import numpy as np

def n_step_G(rewards, n_steps, discount_rate):
    lambdas = np.array([discount_rate ** i for i in range(n_steps)])
    G = [rewards[i:min(i+n_steps, len(rewards))] for i in range(len(rewards))]
    G = [el*lambdas[:len(el)] for el in G]
    G = [sum(el) for el in G]
    return G

def discretize(obs, range):
    onehot = tf.one_hot(np.digitize(obs, range) - 1, len(range)).numpy()
    return(onehot)

def mlp(input_shape, sizes, activation=tf.nn.tanh, output_activation=None, dropout=None):
    input = tf.keras.Input(input_shape)
    if dropout is None:  dropout = [None]*len(sizes)

    # Build a feedforward neural network.
    for i, size in enumerate(sizes):
        if len(sizes) > 1 and i == 0:
            x = tf.keras.layers.Dense(units=size, activation=activation)(input)
        elif len(sizes) == 1:
            x = tf.keras.layers.Dense(units=size, activation=output_activation)(input)
        else:
            activation = activation if i != len(sizes)-1 else output_activation
            x = tf.keras.layers.Dense(units=size, activation=activation)(x)

        if dropout[i] is not None:
            x = tf.keras.layers.Dropout(dropout[i])(x)

    model = tf.keras.Model(inputs=input, outputs=x)

    return model

def vpg_loss(weights, actions, logits, n_acts, baselines = None):
    action_masks = tf.one_hot(actions, n_acts)
    log_probs = tf.reduce_sum(tf.squeeze(action_masks)*tf.nn.log_softmax(tf.squeeze(logits)), axis=1)
    if baselines is not None:
        weights = tf.subtract(weights, baselines)
    loss = -tf.reduce_mean(tf.multiply(weights, log_probs))
    return loss

def rewards_to_go(rews, discount_rate = 1):
    rtg = [0]*len(rews)
    for i in reversed(range(len(rews))):
        rtg[i] = rews[i] + (discount_rate*rtg[i+1] if i+1 < len(rews) else 0)
    return rtg