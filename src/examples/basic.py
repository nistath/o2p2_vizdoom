#!/usr/bin/env python3

from vizdoom import *
import random
import time
import os

game = DoomGame()
game.load_config(os.path.join(scenarios_path, "basic.cfg"))
game.init()

shoot = [False, False, True]
left = [True, False, False]
right = [False, True, False]
actions = [shoot, left, right]

episodes = 10
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer
        misc = state.game_variables
        reward = game.make_action(random.choice(actions))
        print("\treward:", reward)
    print("Result:", game.get_total_reward())
    time.sleep(2)
