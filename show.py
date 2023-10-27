import numpy as np
import pickle
import sys
from time import perf_counter as tpc


from plot import plot_random_small_hist
from utils import Log
from utils import tex_auto_end
from utils import tex_auto_start
from utils import tex_err_val
from utils import tex_multirow
from utils import tex_row_end
from utils import tex_row_line


def show_function_big():
    t_full = tpc()

    log = Log('result/logs_show/function_big.txt')
    log(f'---> SHOW | function_big | \n')

    fpath = 'result/data/function_big.npz'
    result = np.load(fpath, allow_pickle=True)

    data = result.get('data').item()
    d = result.get('d').item()
    n = result.get('n').item()

    text =   '% INFO   > \n'
    text += f'% dim    : {d} \n'
    text += f'% size   : {n} \n'
    text = tex_auto_start(text)

    for name, item in data.items():
        text += name + ' ' * max(0, 15-len(name))
        text += f'& {item["r"]:-5.1f} '
        text += tex_err_val(item['e_val'], '& ')
        text += f'& {item["t"]:-5.1f} '
        text += tex_row_end()
        text += tex_row_line()

    text += tex_auto_end()

    log('\n\n' + text + '\n\n')

    t_full = tpc() - t_full
    log(f'\n===> DONE | function_big | Time: {t_full:-10.3f}\n')


def show_function_small():
    t_full = tpc()

    log = Log('result/logs_show/function_small.txt')
    log('---> SHOW | function_small | \n')

    result = np.load('result/data/function_small.npz', allow_pickle=True)
    data = result.get('data').item()
    d = result.get('d').item()
    n = result.get('n').item()

    text =   '% INFO   > \n'
    text += f'% dim    : {d} \n'
    text += f'% size   : {n} \n'
    text = tex_auto_start(text)

    for name, item in data.items():
        text += name + ' ' * max(0, 15-len(name))
        text += f'& {item["r"]:-5.1f} '
        text += tex_err_val(item['e_min'], '& ')
        text += tex_err_val(item['e_max'], '& ')
        text += tex_row_end()
        text += tex_row_line()

    text += tex_auto_end()

    log('\n\n' + text + '\n\n')

    t_full = tpc() - t_full
    log(f'\n===> DONE | function_small | Time: {t_full:-10.3f}\n')


def show_random_small():
    t_full = tpc()

    log = Log('result/logs_show/random_small.txt')
    log('---> SHOW | random_small | \n')

    result = np.load('result/data/random_small.npz', allow_pickle=True)
    data = result.get('data').item()
    d_ = result.get('d_')
    n_ = result.get('n_')
    r_ = result.get('r_')
    rep = result.get('rep').item()

    num = r_[1] - r_[0] + 1

    text =   '% INFO   > \n'
    text += f'% dims   : {d_[0]} ... {d_[1]} \n'
    text += f'% sizes  : {n_[0]} ... {n_[1]} \n'
    text += f'% ranks  : {r_[0]} ... {r_[1]} \n'
    text += f'% reps   : {rep} \n'
    text = tex_auto_start(text)

    for d in data.keys():
        text += tex_multirow(d, num)

        for r in data[d].keys():
            text += f'& {r:-4d} '
            text += tex_err_val(data[d][r]['e_min'], '& ')
            text += tex_err_val(data[d][r]['e_max'], '& ')
            text += tex_row_end()

        text += tex_row_line()

    text += tex_auto_end()

    log('\n\n' + text + '\n\n')

    t_full = tpc() - t_full
    log(f'\n===> DONE | random_small | Time: {t_full:-10.3f}\n')


def show_random_small_hist(fpath='./result/plot/random_small_hist.png'):
    t_full = tpc()

    log = Log('result/logs_show/random_small_hist.txt')
    log('---> SHOW | random_small_hist | \n')

    result = np.load('result/data/random_small_hist.npz', allow_pickle=True)
    data = result.get('data').item()

    plot_random_small_hist(data, fpath)

    log(f'\n\nPlot is saved into "{fpath}"\n\n')

    t_full = tpc() - t_full
    log(f'\n===> DONE | random_small_hist | Time: {t_full:-10.3f}\n')


if __name__ == '__main__':
    np.random.seed(42)

    mode = sys.argv[1] if len(sys.argv) > 1 else None

    if mode == 'function_big':
        show_function_big()
    elif mode == 'function_small':
        show_function_small()
    elif mode == 'random_small':
        show_random_small()
    elif mode == 'random_small_hist':
        show_random_small_hist()
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
