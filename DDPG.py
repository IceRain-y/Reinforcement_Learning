"""
Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
tensorflow >= 1.0.1
"""

import tensorflow as tf
import numpy as np
import os
import shutil
from arm_env import ArmEnv

tf.compat.v1.disable_eager_execution()

np.random.seed(1)
tf.compat.v1.set_random_seed(1)

MAX_EPISODES = 600
MAX_EP_STEPS = 200
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000
MEMORY_CAPACITY = 5000
BATCH_SIZE = 16
VAR_MIN = 0.1
RENDER = True
LOAD = False
MODE = ['easy', 'hard']
n_model = 1

env = ArmEnv(mode=MODE[n_model])
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound

# all placeholder for tf
with tf.compat.v1.name_scope('S'):
    S = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, STATE_DIM], name='s')
with tf.compat.v1.name_scope('R'):
    R = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1], name='r')
with tf.compat.v1.name_scope('S_'):
    S_ = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, STATE_DIM], name='s_')


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.compat.v1.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
        self.replace = [tf.compat.v1.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s,scope, trainable=True):
        with tf.name_scope(scope):
            init_w = tf.keras.initializers.GlorotUniform()
            init_b = tf.keras.initializers.Constant(0.001)

            # inputs = tf.keras.Input(shape=(s,))  # 假设输入s是特征向量的维度
            net = tf.keras.layers.Dense(200, activation='relu6',
                                      kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                      trainable=trainable)(s)
            net = tf.keras.layers.Dense(200, activation='relu6',
                                      kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                      trainable=trainable)(net)
            net = tf.keras.layers.Dense(10, activation='relu',
                                      kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                      trainable=trainable)(net)
            with tf.name_scope('a'):
                actions = tf.keras.layers.Dense(self.a_dim, activation='tanh',
                                                kernel_initializer=init_w, name='a',
                                                trainable=trainable)(net)
                scaled_a = tf.keras.layers.Lambda(lambda x: x * self.action_bound, name='scaled_a')(actions)

        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run(self.replace)
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.compat.v1.variable_scope('policy_grads'):
            self.policy_grads = tf.compat.v1.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.compat.v1.variable_scope('A_train'):
            opt = tf.compat.v1.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.compat.v1.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.name_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.name_scope('TD_error'):
            self.loss = tf.compat.v1.reduce_mean(tf.compat.v1.squared_difference(self.target_q, self.q))

        with tf.name_scope('C_train'):
            self.train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.name_scope('a_grad'):
            self.a_grads = tf.compat.v1.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)
        self.replace = [tf.compat.v1.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            # init_w = tf.compat.v1.contrib.layers.xavier_initializer()
            init_w = tf.keras.initializers.GlorotUniform()
            #init_b = tf.compat.v1.constant_initializer(0.01)
            init_b = tf.keras.initializers.Constant(0.01)

            with tf.compat.v1.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.compat.v1.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.compat.v1.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.compat.v1.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.compat.v1.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.compat.v1.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.compat.v1.variable_scope('q'):
                q = tf.compat.v1.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run(self.replace)
        self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


sess = tf.compat.v1.Session()

# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.compat.v1.train.Saver()
path = './'+MODE[n_model]

if LOAD:
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint(path))
else:
    sess.run(tf.compat.v1.global_variables_initializer())


def train():
    var = 2.  # control exploration

    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        for t in range(MAX_EP_STEPS):
        # while True:
            if RENDER:
                env.render()

            # Added exploration noise
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)    # add randomness to action selection for exploration
            s_, r, done = env.step(a)
            M.store_transition(s, a, r, s_)

            if M.pointer > MEMORY_CAPACITY:
                var = max([var*.9999, VAR_MIN])    # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_reward += r

            if t == MAX_EP_STEPS-1 or done:
            # if done:
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Explore: %.2f' % var,
                      )
                break

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join('./'+MODE[n_model], 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)


def eval():
    env.set_fps(30)
    s = env.reset()
    while True:
        if RENDER:
            env.render()
        a = actor.choose_action(s)
        s_, r, done = env.step(a)
        s = s_

if __name__ == '__main__':
    if LOAD:
        eval()
    else:
        train()
