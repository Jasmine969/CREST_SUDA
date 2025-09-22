import numpy as np
import pandas as pd


def color_print(text: str, mode: str = '', fore: str = '', back: str = '') -> None:
    dict_mode = {'d': '0', 'h': '1', 'nb': '22', 'u': '4', 'nu': '24',
                 't': '5', 'nt': '25', 'r': '7', 'nr': '27', '': ''}
    dict_fore = {'k': '30', 'r': '31', 'g': '32', 'y': '33', 'b': '34',
                 'm': '35', 'c': '36', 'w': '37', '': ''}
    dict_back = {'k': '40', 'r': '41', 'g': '42', 'y': '43', 'b': '44',
                 'm': '45', 'c': '46', 'w': '47', '': ''}
    formats = ';'.join([each for each in [
        dict_mode[mode], dict_fore[fore], dict_back[back]] if each])
    print(f'\033[{formats}m{text}\033[0m')


def sigmoid(x):
    y = 1 / (1 + np.exp(-np.abs(x)))
    return (1 - y) * (x < 0) + y * (x >= 0)


def save_atomfile(ids, prop_per_atom, save_path='./atomfile'):
    n_atoms = prop_per_atom.size
    df = pd.DataFrame({'ID': ids, 'prop': prop_per_atom})
    df.to_csv(f'{save_path}.csv',
              index=False, sep='\t', header=[str(n_atoms), ''])


def format_time(second):
    second = int(second)
    days = second // 86400
    remaining_seconds = second % 86400
    hours = remaining_seconds // 3600
    remaining_seconds %= 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    return f"{days}d:{hours:02d}:{minutes:02d}:{seconds:02d}"


if __name__ == '__main__':
    print(format_time(3600.1 * 25))
