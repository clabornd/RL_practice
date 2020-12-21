import tensorflow as tf

class acrobot_mlp(tf.keras.Model):
  def __init__(self, sizes, n_acts, activations, output_activation=None, **kwargs):
    super().__init__()
    self.linears = [tf.keras.layers.Dense(size, activation=activation, **kwargs) \
                    for size, activation in zip(sizes, activations)]
    self.dense_out = tf.keras.layers.Dense(n_acts, output_activation, use_bias=False)

  def call(self, inputs):
    # hidden activations for the 1-variable and 2-variable inputs
    hiddens = [l(x) for l,x in zip(self.linears, inputs)]

    # list of the hidden activations for 1 and 2 variable inputs, and the raw tilings of 3 and 4 variable input
    x = [hiddens[i] if i < len(hiddens) else inputs[i] for i in range(len(inputs))]

    # we concatenate the hidden and raw tilings and send them through a final linear layer.
    return self.dense_out(tf.concat(x, axis=1))
