#!/usr/bin/env python3

import vizdoom as vzd

from tqdm import tqdm
import random
import pickle
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt

DEFAULT_CONFIG = str(Path(vzd.scenarios_path).joinpath('basic.cfg'))
if __name__ == '__main__':
    parser = ArgumentParser('Collect experience.')
    parser.add_argument('--config',
                        default=DEFAULT_CONFIG,
                        type=str,
                        help='Path to the configuration file of the scenario.')
    parser.add_argument('out_dir',
                        type=Path)
    parser.add_argument('episodes',
                        default=10,
                        type=int)

    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)

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
        recpath = args.out_dir.joinpath(f'episode_{i}.lmp')
        game.new_episode(str(recpath))
        while not game.is_episode_finished():
            state = game.get_state()
            game.make_action(random.choice(actions))

    game.close()
