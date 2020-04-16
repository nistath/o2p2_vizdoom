import vizdoom as vzd

__all__ = ['game_variables']

game_variables = [getattr(vzd.GameVariable, x) for x in dir(vzd.GameVariable) if x[0] != '_']
