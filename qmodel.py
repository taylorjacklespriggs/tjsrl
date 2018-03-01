import numpy as np
import tensorflow as tf

class QModel:
    def __init__(self, session, num_features, num_actions, parameters):
        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_rate = np.exp(-parameters['log_learning_rate'])
        self.num_hidden_layer_1 = parameters['layer_1_size']
        self.num_hidden_layer_2 = parameters['layer_2_size']
        self.X = tf.placeholder("float", [None, self.num_features])
        self.Y = tf.placeholder("float", [None, self.num_actions])
        self.weights = {
            'h1': tf.Variable(tf.random_normal([
                self.num_features,
                self.num_hidden_layer_1
            ])),
            'h2': tf.Variable(tf.random_normal([
                self.num_hidden_layer_1,
                self.num_hidden_layer_2
            ])),
            'out': tf.Variable(tf.random_normal([
                self.num_hidden_layer_2,
                self.num_actions
            ])),
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.num_hidden_layer_1])),
            'b2': tf.Variable(tf.random_normal([self.num_hidden_layer_2])),
            'out': tf.Variable(tf.random_normal([self.num_actions])),
        }
        self.layer_1 = tf.add(
            tf.matmul(self.X, self.weights['h1']),
            self.biases['b1']
        )
        self.layer_2 = tf.add(
            tf.matmul(self.layer_1, self.weights['h2']),
            self.biases['b2']
        )
        self.out_layer = tf.matmul(self.layer_2, self.weights['out']) + self.biases['out']
        self.loss_op = tf.nn.l2_loss(self.out_layer - self.Y)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        self.session = session
        init = tf.global_variables_initializer()
        self.session.run(init)

    def feed_one(self, state):
        state = state.reshape((1, self.num_features))
        return self.out_layer.eval({self.X: state}, session=self.session)[0]

    def feed(self, states):
        return self.out_layer.eval({self.X: states}, session=self.session)

    def train(self, batch):
        batch_x, batch_y = batch
        _, c = self.session.run([self.train_op, self.loss_op], feed_dict={
            self.X: batch_x,
            self.Y: batch_y,
        })

    def loss(self, batch):
        batch_x, batch_y = batch
        return self.loss_op.eval({self.X: batch_x, self.Y: batch_y})

    def max_loss(self, batch):
        batch_x, batch_y = batch
        return tf.argmax(tf.square(self.out_layer - self.Y), 1).eval({self.X: batch_x, self.Y: batch_y}, session=self.session)
