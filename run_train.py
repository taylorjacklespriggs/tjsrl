import numpy as np
import tensorflow as tf
from time import time
from ll import LunarLander
from qmodel import QModel
from sample_store import SampleStore

BATCH_SIZE = 2**8
EXTRA_STORE = 2**4
TRAINING_ITERATIONS = 500

def argmax(l):
    return max(enumerate(l), key=lambda v: v[1])[0]

class Trainer:
    def __init__(self, game, session, parameters):
        self.qmodel = QModel(session, 8, 4, parameters)
        self.sample_store = SampleStore(EXTRA_STORE * BATCH_SIZE)
        self.game = game
        self.last_state = None
        self.estimate_q = None
        self.max_cumulative = None
        self.prob_scale = np.exp(-parameters['log_prob_scale'])
        self.discount = 1 - np.exp(-parameters['log_discount'])

    def reset(self):
        self.last_state = self.game.reset()
        self.estimate_q = self.qmodel.feed_one(self.last_state)

    def update(self, state):
        self.last_state = state
        self.estimate_q = self.qmodel.feed_one(state)

    def get_action(self):
        logits = np.exp(
            (self.estimate_q - self.estimate_q.max()) * self.prob_scale
        )
        logits /= logits.sum()
        action = np.random.choice(np.arange(4, dtype=np.int8), p=logits)
        return action, self.estimate_q[action]

    def sample(self):
        action, action_q = self.get_action()
        state, reward, done = self.game.do_action(action)
        self.sample_store.add_sample(self.last_state, action, state, reward)
        self.update(state)
        return reward, done

    def compute_update(self):
        state, action, new_state, reward = self.sample_store.get_batch(BATCH_SIZE)
        q = self.qmodel.feed(state)
        q[:, action] = reward
        q_update = self.discount * self.qmodel.feed(new_state).max(axis=1)
        if self.max_cumulative is None:
            q[:, action] += q_update
        else:
            mx = np.full((action.shape[0], 1), self.max_cumulative)
            mask = q_update > mx
            q[:, action] += np.where(mask, self.max_cumulative, q_update)
        return state, q

    def train(self):
        self.qmodel.train(self.compute_update())

    def show(self):
        self.reset()
        self.game.render()
        done = False
        while not done:
            action, _ = self.get_action()
            state, _, done = self.game.do_action(action)
            self.update(state)
            game.render()

    def evaluate(self):
        print("Evaluating")
        state, action, new_state, reward = self.sample_store.get_all()
        q = self.qmodel.feed(state)
        q[:, action] = reward
        q[:, action] += self.qmodel.feed(new_state).max(axis=1)
        return np.log(self.qmodel.loss((state, q)))

    def run(self):
        i = 0
        r = 10
        render = 10
        for _ in range(TRAINING_ITERATIONS):
            self.reset()
            print("Game {:5d}".format(i), end='\r')
            i += 1
            done = False
            cumulative = 0
            while not done:
                reward, done = self.sample()
                self.train()
                cumulative += reward
                if r == render:
                    self.game.render(0)
            if self.max_cumulative is None:
                self.max_cumulative = cumulative
            else:
                self.max_cumulative = max(self.max_cumulative, cumulative)
            if r >= render:
                r = 0
            r += 1
        print()
        print("Finished training")

if __name__ == '__main__':
    from sys import argv, exit
    if len(argv) <= 1 or argv[1] not in ('training', 'evaluating'):
        print("training or evaluating")
        exit(1)
    game = LunarLander()
    import sigopt
    if argv[1] == 'training':
        connection = sigopt.Connection()
        e = connection.experiments().create(
            name="RL LL {}".format(time()),
            parameters=[
                dict(
                    name='log_discount',
                    bounds=dict(min=0, max=6),
                    type='double',
                ),
                dict(
                    name='log_learning_rate',
                    bounds=dict(min=3, max=15),
                    type='double',
                ),
                dict(
                    name='layer_1_size',
                    bounds=dict(min=1, max=200),
                    type='int',
                ),
                dict(
                    name='layer_2_size',
                    bounds=dict(min=1, max=200),
                    type='int',
                ),
                dict(
                    name='log_prob_scale',
                    bounds=dict(min=0, max=10),
                    type='double',
                ),
            ]
        )
        with tf.Session() as session:
            while True:
                suggestion = connection.experiments(e.id).suggestions().create()
                assignments = suggestion.assignments
                trainer = Trainer(game, session, assignments)
                trainer.run()
                evaluation = trainer.evaluate()
                connection.experiments(e.id).observations().create(
                    suggestion=suggestion.id,
                    values=[dict(value=-float(evaluation))]
                )
    else:
        with tf.Session() as session:
            assignments = {
                'layer_1_size': 166,
                'layer_2_size': 75,
                'log_discount': 1.0665122459164607,
                'log_learning_rate': 13.334311180363606,
                'log_prob_scale': 10.0,
            }
            trainer = Trainer(game, session, assignments)
            trainer.run()
