#!/usr/bin/env python3

from random import choice
import vizdoom as vzd
from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt

def print_all(obj):
    d = {}
    for attr in dir(obj):
        if attr.startswith('_'):
            continue

        d[attr] = getattr(obj, attr)

    print(type(obj).__name__, d)

def print_iter(iterable):
    print('[')
    for x in iterable:
        print_all(x)

    print(']')


DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "basic.cfg")
if __name__ =="__main__":
    parser = ArgumentParser("ViZDoom example showing how to use information about objects and map.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")

    args = parser.parse_args()

    game = vzd.DoomGame()

    # Use other config file if you wish.
    game.load_config(args.config)
    game.set_render_hud(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Enables information about all objects present in the current episode/level.
    game.set_objects_info_enabled(True)

    # Enables information about all sectors (map layout).
    game.set_sectors_info_enabled(True)

    game.set_labels_buffer_enabled(True)

    game.clear_available_game_variables()
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Z)

    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]

    episodes = 10
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        while not game.is_episode_finished():
            # Gets the state
            state = game.get_state()
            game.make_action(choice(actions))

            print_iter(state.objects)
            print_iter(state.labels)
            print(state.labels_buffer.shape)
            plt.imshow(state.labels_buffer)
            plt.show()


        print("Episode finished!")

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()
