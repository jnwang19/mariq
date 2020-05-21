__author__ = 'justinarmstrong'

import os
import pygame as pg
import numpy as np
from . import DQN
from random import randint
from keras.utils import to_categorical

keybinding = {
    'action':pg.K_s,
    'jump':pg.K_a,
    'left':pg.K_LEFT,
    'right':pg.K_RIGHT,
    'down':pg.K_DOWN
}

movebinding = [pg.K_s, pg.K_a, pg.K_LEFT, pg.K_RIGHT, pg.K_DOWN]

def define_parameters():
    params = dict()
    params['epsilon_decay_linear'] = 1/75
    params['learning_rate'] = 0.0005
    params['first_layer_size'] = 150   # neurons in the first layer
    params['second_layer_size'] = 150   # neurons in the second layer
    params['third_layer_size'] = 150    # neurons in the third layer
    params['episodes'] = 50            
    params['memory_size'] = 2500
    params['batch_size'] = 500
    params['weights_path'] = 'weights/weights.hdf5'
    params['load_weights'] = True
    params['train'] = False
    return params

class Control(object):
    """Control class for entire project. Contains the game loop, and contains
    the event_loop which passes events to States as needed. Logic for flipping
    states is also found here."""
    def __init__(self, caption):
        self.screen = pg.display.get_surface()
        self.done = False
        self.clock = pg.time.Clock()
        self.caption = caption
        self.fps = 60
        self.show_fps = True
        self.current_time = 0.0
        self.keys = pg.key.get_pressed()
        self.state_dict = {}
        self.state_name = None
        self.state = None
        self.display = True

    def setup_states(self, state_dict, start_state):
        self.state_dict = state_dict
        self.state_name = start_state
        self.state = self.state_dict[self.state_name]

    def update(self):
        self.current_time = pg.time.get_ticks()
        if self.state.quit:
            self.done = True
        elif self.state.done:
            self.flip_state()
        self.state.update(self.screen, self.keys, self.current_time)

    def flip_state(self):
        previous, self.state_name = self.state_name, self.state.next
        persist = self.state.cleanup()
        self.state = self.state_dict[self.state_name]
        self.state.startup(self.current_time, persist)
        self.state.previous = previous


    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
            elif event.type == pg.KEYDOWN:
                self.keys = pg.key.get_pressed()
                self.toggle_show_fps(event.key)
            elif event.type == pg.KEYUP:
                self.keys = pg.key.get_pressed()
            self.state.get_event(event)


    def toggle_show_fps(self, key):
        if key == pg.K_F5:
            self.show_fps = not self.show_fps
            if not self.show_fps:
                pg.display.set_caption(self.caption)


    def main(self):
        """Main loop for entire program"""
        params = define_parameters()
        agent = DQN.DQNAgent(params)
        weights_filepath = params['weights_path']
        if params['load_weights']:
            agent.model.load_weights(weights_filepath)
            print("weights loaded")
        counter_games = 0
        crash = False
        new_game = False # flag to signify a new game has started

        while not self.done and counter_games < params['episodes']:
            while not crash:
                self.event_loop()
                self.update()
                if self.display:
                    pg.display.update()
                self.clock.tick(self.fps)
                if self.show_fps:
                    fps = self.clock.get_fps()
                    with_fps = "{} - {:.2f} FPS".format(self.caption, fps)
                    pg.display.set_caption(with_fps)
                if self.state_name == 'level1':
                    if not self.state.mario.dead:
                        new_game = True

                    if not params['train']:
                        agent.epsilon = 0
                    else:
                        # agent.epsilon is set to give randomness to actions
                        agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

                    state_old = np.asarray(self.state.mario.rect.x // 20)
                        
                    # perform random actions based on agent.epsilon, or choose the action
                    if randint(0, 1) < agent.epsilon:
                        final_move = np.random.randint(2, size=5)
                    else:
                        # predict action based on the old state
                        prediction = agent.model.predict(state_old.reshape((1, 1)))
                        choice = np.argmax(prediction[0])
                        if randint(0, 1) < 0.1:
                            choice = np.random.choice(5, p=prediction[0])
                        final_move = to_categorical(choice, num_classes=5)

                    # perform new move and get new state
                    self.keys = self.convert_to_keys(final_move)
                    state_new = np.asarray(self.state.mario.rect.x // 20)

                    # set reward for the new state
                    reward = self.set_reward(state_old, state_new, self.state.mario.dead)

                    if params['train']:
                        # train short memory base on the new action and state
                        agent.train_short_memory(state_old, final_move, reward, state_new)
                        # store the new data into a long term memory
                        agent.remember(state_old, final_move, reward, state_new)
                        
                    if self.state.mario.dead:
                        crash = True

            if params['train'] and new_game:
                new_game = False
                agent.replay_new(agent.memory, params['batch_size'])
                counter_games += 1
            crash = False

        if params['train']:
            agent.model.save_weights(params['weights_path'])
            print(agent.model.get_weights())
            print('saved')
            

    def convert_to_keys(self, moveVec):
        keys = [0] * 323 # normal pg.keys is length 323
        for i in range(len(moveVec)):
            if moveVec[i] == 1:
                keys[movebinding[i]] = 1
        return keys

    def set_reward(self, state_old, state_new, is_dead):
        if is_dead:
            return -10
        if state_new > state_old:
            return 1
        elif state_new <= state_old:
            return -1
        return 0

class _State(object):
    def __init__(self):
        self.start_time = 0.0
        self.current_time = 0.0
        self.done = False
        self.quit = False
        self.next = None
        self.previous = None
        self.persist = {}

    def get_event(self, event):
        pass

    def startup(self, current_time, persistant):
        self.persist = persistant
        self.start_time = current_time

    def cleanup(self):
        self.done = False
        return self.persist

    def update(self, surface, keys, current_time):
        pass



def load_all_gfx(directory, colorkey=(255,0,255), accept=('.png', 'jpg', 'bmp')):
    graphics = {}
    for pic in os.listdir(directory):
        name, ext = os.path.splitext(pic)
        if ext.lower() in accept:
            img = pg.image.load(os.path.join(directory, pic))
            if img.get_alpha():
                img = img.convert_alpha()
            else:
                img = img.convert()
                img.set_colorkey(colorkey)
            graphics[name]=img
    return graphics


def load_all_music(directory, accept=('.wav', '.mp3', '.ogg', '.mdi')):
    songs = {}
    for song in os.listdir(directory):
        name,ext = os.path.splitext(song)
        if ext.lower() in accept:
            songs[name] = os.path.join(directory, song)
    return songs


def load_all_fonts(directory, accept=('.ttf')):
    return load_all_music(directory, accept)


def load_all_sfx(directory, accept=('.wav','.mpe','.ogg','.mdi')):
    effects = {}
    for fx in os.listdir(directory):
        name, ext = os.path.splitext(fx)
        if ext.lower() in accept:
            effects[name] = pg.mixer.Sound(os.path.join(directory, fx))
    return effects











