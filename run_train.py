import numpy as np
import tensorflow as tf
from time import time
from ll import LunarLander
from qmodel import QModel
from sample_store import SampleStore
from keyboard_input import KeyboardInput
from get_samples import get_samples

BATCH_SIZE = 2**8
EXTRA_STORE = 2**6
TRAINING_ITERATIONS = 2**16

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
        self.prob_scale = 1 #np.exp(-parameters['log_prob_scale'])
        self.discount = 1 - np.exp(-parameters['log_discount'])

    def reset(self):
        self.last_state = self.game.reset()
        self.estimate_q = self.qmodel.feed_one(self.last_state)

    def update_with_sample(self, action, state, reward):
        self.sample_store.add_sample(self.last_state, action, state, reward)
        self.update(state)

    def update(self, state):
        self.last_state = state
        self.estimate_q = self.qmodel.feed_one(state)

    def get_action(self):
        logits = np.exp(
            (self.estimate_q - self.estimate_q.max()) * self.prob_scale
        )
        logits /= logits.sum()
        action = np.random.choice(np.arange(4, dtype=np.int8), p=logits)
        return action


    def sample(self, action):
        state, reward, done = self.game.do_action(action)
        self.update_with_sample(action, state, reward)
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
            action = self.get_action()
            state, _, done = self.game.do_action(action)
            self.update(state)
            game.render()

    def evaluate(self):
        print("Evaluating")
        state, action, new_state, reward = self.sample_store.get_all()
        q = self.qmodel.feed(state)
        q[:, action] = reward
        q[:, action] += self.qmodel.feed(new_state).max(axis=1)
        return np.log(self.qmodel.loss((state, q)) / state.shape[0])

    def _run_one(self, get_action):
        self.reset()
        done = False
        cumulative = 0
        while not done:
            reward, done = self.sample(get_action())
            self.train()
            cumulative += reward
        if self.max_cumulative is None:
            self.max_cumulative = cumulative
        else:
            self.max_cumulative = max(self.max_cumulative, cumulative)
        return self.evaluate(), cumulative

    def _run(self, get_action):
        return iter(lambda: self._run_one(get_action), None)

    def run_one_with_input(self):
        ki = KeyboardInput()
        ki.add_to_window(game.env.unwrapped.viewer.window)
        def get_action():
            self.game.render()
            return ki.get_action()
        return self._run_one(get_action)

    def run_with_input(self):
        ki = KeyboardInput()
        ki.add_to_window(game.env.unwrapped.viewer.window)
        def get_action():
            self.game.render()
            return ki.get_action()
        for loss, cumulative in self._run(get_action):
            yield loss, cumulative

    def run(self, training_iterations=TRAINING_ITERATIONS):
        for _, (loss, cumulative) in zip(range(training_iterations), self._run(self.get_action)):
            yield loss, cumulative

    def run_alternating(self):
        for _ in self.run_with_input(): pass
        self.show()
        self.run()
        self.show()

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
                    name='log_discount',
                    bounds=dict(min=0, max=6),
                    type='double',
                ),
                dict(
                    name='log_learning_rate',
                    bounds=dict(min=3, max=15),
                    type='double',
                ),
            ]
        )
        with tf.Session() as session:
            import json
            with open('assignments.json') as assignments_file:
                assignments = json.load(assignments_file)
            trainer = Trainer(game, session, assignments)
            for loss, cumulative in trainer.run(10**6):
                print(cumulative)
            sample_store = trainer.sample_store
            while True:
                suggestion = connection.experiments(e.id).suggestions().create()
                assignments = suggestion.assignments
                trainer = Trainer(game, session, assignments)
                trainer.sample_store = sample_store
                for i in range(TRAINING_ITERATIONS // 100):
                    print(i * 100, end='\r')
                    for _ in range(100):
                        trainer.train()
                loss = trainer.evaluate()
                print(loss)
                connection.experiments(e.id).observations().create(
                    suggestion=suggestion.id,
                    values=[dict(value=-float(loss))]
                )
            #     suggestion = connection.experiments(e.id).suggestions().create()
            #     assignments = suggestion.assignments
            #     trainer = Trainer(game, session, assignments)
            #     trainer.run()
            #     evaluation = trainer.evaluate()
            #     connection.experiments(e.id).observations().create(
            #         suggestion=suggestion.id,
            #         values=[dict(value=-float(evaluation))]
            #     )
    else:
        import json
        with open('assignments.json') as assignments_file:
            assignments = json.load(assignments_file)
        with tf.Session() as session:
            trainer = Trainer(game, session, assignments)
            trainer.run_alternating()
