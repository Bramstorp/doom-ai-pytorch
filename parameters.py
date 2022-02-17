class Parameters():
    def __init__(self):
        self.scenario = 'deadly_corridor'
        self.learning_rate = 0.00025
        self.gamma = 0.99
        self.train_epochs = 10
        self.num_steps = 2000
        self.model = 'a3c'
        self.replay_memory_size = 10000
        self.batch_size = 64
        self.frame_repeat = 12