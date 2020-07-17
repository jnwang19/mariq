from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add
import collections
from .components.enemies import Goomba

class DQNAgent(object):
    def __init__(self, params):
        self.reward = 0
        self.gamma = 0.999
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']        
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.input_dim = 5
        self.model = self.network()

    def network(self):
        model = Sequential()
        model.add(Dense(output_dim=self.first_layer, activation='relu', input_dim=self.input_dim))
        model.add(Dense(output_dim=self.second_layer, activation='relu'))
        model.add(Dense(output_dim=self.third_layer, activation='relu'))
        model.add(Dense(output_dim=12, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if self.load_weights:
            model.load_weights(self.weights)
        return model
    
    def get_state(self, mario, sprites):
        state = [0] * 5

        # mario mode
        if mario.big:
            state[0] = 1
        if mario.fire:
            state[0] = 2

        # direction
        if mario.x_vel < 0 and mario.y_vel < 0:
            state[1] = 0
        elif mario.x_vel == 0 and mario.y_vel < 0:
            state[1] = 1
        elif mario.x_vel > 0 and mario.y_vel < 0:
            state[1] = 2
        elif mario.x_vel < 0 and mario.y_vel == 0:
            state[1] = 3
        elif mario.x_vel == 0 and mario.y_vel == 0:
            state[1] = 4
        elif mario.x_vel > 0 and mario.y_vel == 0:
            state[1] = 5
        elif mario.x_vel < 0 and mario.y_vel > 0:
            state[1] = 6
        elif mario.x_vel == 0 and mario.y_vel > 0:
            state[1] = 7
        else:
            state[1] = 8

        # enemies
        for sprite in sprites:
            if type(sprite) is Goomba:
                if sprite.rect.x > mario.rect.x:
                    state[2] = 1
                elif sprite.rect.x < mario.rect.x:
                    state[3] = 1

        # can jump
        if mario.allow_jump:
            state[4] = 1

        return np.asarray(state)

    def set_reward(self, mario_old, mario_new, is_dead, passed_checkpoint, win):
        if win:
            return 10000
        x_old = mario_old[0]
        x_new = mario_new[0]
        big_old = mario_old[2]
        big_new = mario_new[2]
        if is_dead:
            return -100
        if passed_checkpoint:
            print('passed checkpoint')
            return 1000
        if big_new and not big_old:
            print('big')
            return 5000
        if x_new == x_old:
            return -10
        return x_new - x_old# + (y_old - y_new) * 0.1

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state in minibatch:
            target = reward
            target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state):
        target = reward
        target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, self.input_dim)))[0])
        target_f = self.model.predict(state.reshape((1, self.input_dim)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, self.input_dim)), target_f, epochs=1, verbose=0)