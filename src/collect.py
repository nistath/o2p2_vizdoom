#!/usr/bin/env python3

import vizdoom as vzd

from tqdm import tqdm
import random
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt

DEFAULT_CONFIG = str(Path(vzd.scenarios_path).joinpath("basic.cfg"))
if __name__ == "__main__":
    parser = ArgumentParser("Collect experience.")
    parser.add_argument("--config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        type=str,
                        help="Path to the configuration file of the scenario.")
    parser.add_argument("episodes",
                        default=10,
                        type=int)

    args = parser.parse_args()

    game = vzd.DoomGame()

    # Use other config file if you wish.
    game.load_config(args.config)
    game.set_render_hud(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Enables information about all objects present in the current episode/level.
    game.set_objects_info_enabled(True)
    # Enables segmentation maps for all objects.
    game.set_labels_buffer_enabled(True)
    # Enables information about all sectors (map layout).
    game.set_sectors_info_enabled(True)

    # Set all game variables
    game.set_available_game_variables(game.get_available_game_variables())

    game.init()

    left = [True, False, False]
    right = [False, True, False]
    shoot = [False, False, True]
    actions = [left, right, shoot]

    for i in tqdm(range(args.episodes)):
        game.new_episode()
        while not game.is_episode_finished():
            # Gets the state
            state = game.get_state()
            game.make_action(random.choice(actions))

    game.close()
