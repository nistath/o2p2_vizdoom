#!/usr/bin/env python3

from vizdoom import *
import random
import time
import os

game = DoomGame()
game.load_config(os.path.join(scenarios_path, "basic.cfg"))
game.init()

shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
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
