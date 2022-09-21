import random
from tensorflow.keras import Model
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import squeeze, expand_dims
import tensorflow.keras.backend as K
from tensorflow.keras import losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Dropout, MaxPool1D
from tensorflow.keras.optimizers import Adam
# from keras_radam.training import RAdamOptimizer
from trading_bot.MultiHeadAttention import MultiHeadSelfAttention
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """ Stock Trading Bot """

    def __init__(self, state_size, strategy="t-dqn", reset_every=10, pretrained=False, model_name=None,
                 memoryLenDefault=1000):
        self.strategy = strategy
        self.memoryLenDefault = memoryLenDefault

        # agent config
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=self.memoryLenDefault)
        self.first_iter = True

        # model config
        self.model_name = model_name
        self.gamma = 0.95  # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(learning_rate=self.learning_rate)

        if pretrained and self.model_name is not None:
            self.model = self._model()
            self.load()
        else:
            self.model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):
        """Creates the model
        """
        input_layer = Input(shape=(self.state_size,))
        h1 = Dense(64, activation="relu")(input_layer)
        h1_normed = BatchNormalization()(h1)
        h2 = Dense(128, activation="relu")(h1_normed)
        h2_normed = BatchNormalization()(h2)
        h3 = Dense(256, activation="relu")(h2_normed)
        h3_normed = BatchNormalization()(h3)

        h4 = squeeze(MultiHeadSelfAttention(4, 256)(h3_normed), 1)
        h4_normed = BatchNormalization()(h4)
        h5 = Dense(256, activation="relu")(h4_normed)
        h5_normed = BatchNormalization()(h5)
        h6 = Dense(128, activation="relu")(h5_normed + h3_normed)
        h6_normed = BatchNormalization()(h6)
        h7 = Dense(64, activation="relu")(h6_normed + h2_normed)
        h7_normed = BatchNormalization()(h7)

        output = Dense(self.action_size)(h7_normed)
        weight = Dense(1, activation='sigmoid')(h7_normed)
        model = Model(input_layer, [output, weight])

        model.compile(loss=[self.loss, self.loss], loss_weights=[1, 1], optimizer=self.optimizer)
        return model

    def reset(self):
        self.memory = deque(maxlen=self.memoryLenDefault)

    def remember(self, state, action, reward, next_state, done, actWeight, data_t):
        """Adds relevant data to memory
        """
        self.memory.append((state, action, reward, next_state, done, actWeight, data_t))

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions
        """
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1, 0  # make a definite buy on the first iter

        action_probs, actWeight = self.model.predict_on_batch(state)
        action = np.argmax(action_probs[0])
        return action, actWeight[0][0]

    def train_experience_replay(self, batch_size, t):
        """Train on previous experiences in memory
        """
        random.shuffle(self.memory)
        mini_batch = list(self.memory)[:batch_size]
        X_train, y_train, X_weights, weights = [], [], [], []

        # DQN
        if self.strategy == "dqn":
            for state, action, reward, next_state, done, x_weights, data_t in mini_batch:
                nextPre = self.model.predict_on_batch(next_state)
                # estimate q-values based on current state
                q_values, weight = nextPre[0], nextPre[1]
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation
                    target = reward + self.gamma * np.amax(nextPre[0][0])

                # update the target for current action based on discounted reward
                q_values[0][action] = target
                # weight = weight + self.gamma * np.amax(nextPre[0][0])/\
                #          (reward+0.001) # avoid error when reward equals 0
                weight = x_weights
                X_train.append(state[0])
                y_train.append(q_values[0])
                X_weights.append(x_weights)
                weights.append(weight)

        else:
            raise NotImplementedError()
        # update parameters based on huber loss and mse
        loss = self.model.train_on_batch(
            [tf.convert_to_tensor(X_train, dtype=np.float32), tf.convert_to_tensor(X_weights, dtype=np.float32)],
            [tf.convert_to_tensor(y_train, dtype=np.float32), tf.convert_to_tensor(weights, dtype=np.float32)],
        )
        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self):
        self.model.save_weights("models/{}.h5".format(self.model_name))

    def load(self):
        self.model.load_weights("models/{}.h5".format(self.model_name))
