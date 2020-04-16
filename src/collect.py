#!/usr/bin/env python3

import vizdoom as vzd

from game_variables import game_variables
from serialize import Saver

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
    parser.add_argument('--record_lmp',
                        dest='record_lmp',
                        action='store_true')
    parser.set_defaults(record_lmp=False)

    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)
    if args.record_lmp:
        lmp_dir = out_dir.joinpath('lmp')
        lmp_dir.mkdir()

    game = vzd.DoomGame()

    # Use other config file if you wish.
    game.load_config(args.config)
    game.set_render_hud(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Episodes can be recorded in any available mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_window_visible(False)

    # Enables information about all objects present in the current episode/level.
    game.set_objects_info_enabled(True)
    # Enables segmentation maps for all objects.
    game.set_labels_buffer_enabled(True)
    # Enables information about all sectors (map layout).
    game.set_sectors_info_enabled(True)

    game.set_depth_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)

    # Set all game variables
    game.set_available_game_variables(game_variables)

    game.init()

    left = [True, False, False]
    right = [False, True, False]
    shoot = [False, False, True]
    actions = [left, right, shoot]

    saver = Saver(game)

    for i in tqdm(range(args.episodes)):
        if args.record_lmp:
            recpath = args.out_dir.joinpath(f'episode_{i}.lmp')
            game.new_episode(str(recpath))
        else:
            game.new_episode()

        while not game.is_episode_finished():
            state = game.get_state()

            saver.add(state, i)
            game.make_action(random.choice(actions))

    saver.save(args.out_dir.joinpath('states.npz'), False)
    game.close()
