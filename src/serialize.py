import vizdoom as vzd

import numpy as np
from pathlib import Path


def lmps_to_dataset(lmp_paths):
    game = vzd.DoomGame()
    game.init()
    game.set_window_visible(False)

    # Enables information about all objects present in the current episode/level.
    game.set_objects_info_enabled(True)
    # Enables segmentation maps for all objects.
    game.set_labels_buffer_enabled(True)
    # Enables information about all sectors (map layout).
    game.set_sectors_info_enabled(True)

    for path in lmp_paths:
        episode_data = []

        while not game.is_episode_finished():
            state = game.get_state()
            print(state.labels_buffer)
            game.advance_action()

            episode_data.append(state.number)

        yield episode_data


def to_dict(obj):
    return {attr: getattr(obj, attr) for attr in dir(obj) if attr[0] != '_'}

def to_dict_foreach(iterable):
    return [to_dict(x) for x in iterable]


class Saver:
    def __init__(self, game):
        GV_LEN = len('GameVariable.')
        game_variables_dtype = [(str(x)[GV_LEN:], np.double)
                                for x in game.get_available_game_variables()]
        buffer_shape = (game.get_screen_height(), game.get_screen_width())
        self.dtype = [
            ('number', np.int),
            ('tic', np.int),
            ('game_variables', np.double, len(game.get_available_game_variables())),
            # ('game_variables', game_variables_dtype),
            ('screen_buffer', np.uint8, (3,) + buffer_shape),
        ]

        self.is_depth_buffer_enabled = game.is_depth_buffer_enabled()
        if self.is_depth_buffer_enabled:
            self.dtype.append(('depth_buffer', np.uint8, buffer_shape))

        self.is_labels_buffer_enabled = game.is_labels_buffer_enabled()
        if self.is_labels_buffer_enabled:
            self.dtype.append(('labels_buffer', np.uint8, buffer_shape))

        self.is_automap_buffer_enabled = game.is_automap_buffer_enabled()
        if self.is_automap_buffer_enabled:
            self.dtype.append(
                ('automap_buffer', np.uint8, (3,) + buffer_shape))

        self.dtype.append(('labels', 'O'))
        self.dtype.append(('objects', 'O'))

        self.dtype = np.dtype(self.dtype)

        self.states = []

    def clear(self):
        self.states = []

    def add(self, state):
        array = [state.number, state.tic,
                 state.game_variables, state.screen_buffer]
        if self.is_depth_buffer_enabled:
            array.append(state.depth_buffer)
        if self.is_labels_buffer_enabled:
            array.append(state.labels_buffer)
        if self.is_automap_buffer_enabled:
            array.append(state.automap_buffer)
        array.append(to_dict_foreach(state.labels))
        array.append(to_dict_foreach(state.objects))

        array = np.array([tuple(array)], dtype=self.dtype)
        self.states.append(array)
        return array

    def save(self, filename):
        stack = np.vstack(self.states)
        np.save(filename, stack)


if __name__ == '__main__':
    print(list(lmps_to_dataset(['/home/nistath/Desktop/run1/episode_1.lmp'])))
