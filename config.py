class Config(object):
    learning_rate = 0.004
    batch_size = 1000
    td_steps = 4
    epochs = 50
    discount_rate = 0.99
    trace_decay = 0.9
    explore_0 = 0.5
    explore_decay = 0.9
    min_explore = 0.001