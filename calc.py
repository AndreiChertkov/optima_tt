import numpy as np
import sys
import teneva
from teneva_bm import *
from time import perf_counter as tpc


from fpe import FpeOUP
from show import show_function_big
from show import show_function_small
from show import show_random_small
from show import show_random_small_hist
from utils import Log
from utils import folder_ensure


BMS_FUNC_SMALL = [
    BmFuncAckley,
    BmFuncAlpine,
    BmFuncChung,
    BmFuncDixon,
    BmFuncExp,
    BmFuncMichalewicz,
    BmFuncPathological,
    BmFuncPinter,
    BmFuncPowell,
    BmFuncQing,
    BmFuncRastrigin,
    BmFuncSchaffer,
    BmFuncSchwefel,
    BmFuncSphere,
    BmFuncSquares,
    BmFuncWavy,
    BmFuncYang]


BMS_FUNC_BIG = [
    BmFuncChung,
    BmFuncExp,
    BmFuncPowell,
    BmFuncRastrigin,
    BmFuncSphere,
    BmFuncSquares,
    BmFuncWavy]


def calc_fpe(d_fpe=3, k=100, m_trn=1.E+5, m_tst=1.E+3, seed=42, offline=True,
              r=25, nswp=15, lamb=1.E-4, noise=1.E-8):
    t_full = tpc()

    solver = FpeOUP(d_fpe, seed)
    solver.prep()
    d = solver.d_opt
    n = solver.n

    log = Log(f'result/logs_calc/fpe{d_fpe}d.txt')
    log(f'---> CALC | fpe{d_fpe}d | d: {d:-3d} | n: {n:-8d} | k: {k:-4d}\n')

    # Generate or load the test dataset:
    fpath_tst = f'result/data/fpe{d_fpe}d_data_tst.npz'
    try:
        I_tst, y_tst, info = solver.data_load(fpath_tst, tst=True)
    except Exception as e:
        I_tst, y_tst, info = solver.data_build(m_tst, fpath_tst, tst=True)
    log(info)

    # Generate or load the train dataset:
    if offline:
        fpath_trn = f'result/data/fpe{d_fpe}d_data_trn.npz'
        try:
            I_trn, y_trn, info = solver.data_load(fpath_trn, tst=False)
        except Exception as e:
            I_trn, y_trn, info = solver.data_build(m_trn, fpath_trn, tst=False)
        log(info)

    t_appr = tpc()

    if offline:
        # Build the TT-approximation by the TT-ANOVA-ALS method:
        info = {}
        Y = teneva.anova(I_trn, y_trn, r=r, order=1, noise=noise, seed=seed)
        Y = teneva.als(I_trn, y_trn, Y, nswp=nswp, e=-1., lamb=lamb,
            I_vld=I_tst, y_vld=y_tst, info=info, log=True)
        e_vld = info['e_vld']

    else:
        # Build the TT-approximation by the TT-CROSS method:
        info = {}
        Y = teneva.rand([n]*d, r=r, seed=seed)
        Y = teneva.cross(solver.run_grid_many, Y, e=1.E-8, m=m_trn, dr_max=0,
            cache={}, I_vld=I_tst, y_vld=y_tst, log=True)
        # Y = teneva.truncate(Y, 1.E-8)
        e_vld = info['e_vld']

    r = teneva.erank(Y)
    t_appr = tpc() - t_appr

    # Find min/max values for TT-tensor by optima_tt:
    t_opti = tpc()
    i_opti = teneva.optima_tt(Y, k)[0]
    t_opti = tpc() - t_opti
    e_opti = solver.run_grid(i_opti)

    text = '\n------------> Resulting indices:\n'
    text += str(i_opti.reshape(solver.d, solver.d)) + '\n\n\n'
    text += f'fpe{d_fpe}d | '
    text += f'r {r:-4.1f} | '
    text += f't_ap {t_appr:-6.2f} | '
    text += f't_op {t_opti:-6.2f} | '
    text += f'e_ap {e_vld:-7.1e} | '
    text += f'e_op {e_opti:-7.1e}'
    log(text)

    # Compute the errors for each time step:
    solver.with_y_list = True
    solver.run(solver.A_opt)
    Y_list_real = [[G.copy() for G in Y] for Y in solver.Y_list]
    A = solver.ind_to_poi(i_opti).reshape(solver.d, solver.d)
    solver.run(A)
    Y_list_calc = [[G.copy() for G in Y] for Y in solver.Y_list]

    text = '\n------------> Errors for time steps:\n'
    for Y_real, Y_calc in zip(Y_list_real, Y_list_calc):
        e = teneva.norm(teneva.sub(Y_real, Y_calc))
        text += f'{e:7.1e}\n'
    log(text + '\n\n')

    t_full = tpc() - t_full
    log(f'\n===> DONE | fpe{d_fpe}d | Time: {t_full:-10.3f}\n\n\n\n')


def calc_function_big(d=100, n=2**10, k=100, mode='tt'):
    t_full = tpc()

    log = Log('result/logs_calc/function_big.txt')
    log(f'---> CALC | function_big | d: {d:-3d} | n: {n:-8d} | k: {k:-4d}\n')

    data = {}

    for Bm in BMS_FUNC_BIG:
        bm = Bm(d, n)
        bm.prep()

        if bm.x_min_real is None or bm.y_min_real is None:
            continue

        # Build the TT-cores:
        Y = bm.build_cores()
        r = teneva.erank(Y)

        # Find the value of the TT-tensor near argmin of the function:
        y_min_real = bm.y_min_real
        x_min_real = bm.x_min_real
        i_min_real = teneva.poi_to_ind(x_min_real,
            bm.a, bm.b, bm.n, bm.grid_kind)
        y_min_appr = bm.get(i_min_real)

        # Find min/max values for TT-tensor by optima_tt:
        t = tpc()
        if mode == 'tt':
            i_min, y_min = teneva.optima_tt(Y, k)[:2]
        else:
            i_min, y_min = teneva.optima_qtt(Y, k)[:2]
        y_min = bm.get(i_min)
        t = tpc() - t

        # Calculate the error:
        e_val = np.abs(y_min - y_min_appr)
        e_ind = np.max(np.abs(i_min - i_min_real))

        name = bm.name
        data[name] = {'t': t, 'r': r, 'e_val': e_val, 'e_ind': e_ind}

        text = ''
        text += name + ' ' * max(0, 18-len(name)) +  ' | '
        text += f'r: {r:-4.1f} | '
        text += f't: {t:-7.3f} | '
        text += f'e_min: {e_val:-7.1e}'

        # NOTE !!!!!
        # If the found by Optima-TT optimum value is better than expected
        # (i.e., in the nearest to the exact global optimimum multi-index of
        # the tensor), this means that the grid discretization is insufficient,
        # and we reasonably skip such benchmarks.
        if y_min < y_min_appr:
            text += f' | BETTER !'

        text += '\n' + ' ' * 18 + ' | ' + f'y_min_real: {y_min_real:-10.3e}'
        text += '\n' + ' ' * 18 + ' | ' + f'y_min_appr: {y_min_appr:-10.3e}'
        text += '\n' + ' ' * 18 + ' | ' + f'y_min_calc: {y_min:-10.3e}'
        log(text)

    fpath = 'result/data/function_big.npz'
    np.savez_compressed(fpath, data=data, d=d, n=n, k=k)

    t_full = tpc() - t_full
    log(f'\n===> DONE | function_big | Time: {t_full:-10.3f}\n\n\n\n')

    show_function_big()


def calc_function_small(d=6, n=20, k=200, seed=42):
    t_full = tpc()

    log = Log('result/logs_calc/function_small.txt')
    log(f'---> CALC | function_small | d: {d:-3d} | n: {n:-8d} | k: {k:-4d}\n')

    data = {}

    for Bm in BMS_FUNC_SMALL:
        bm = Bm(d, n)
        bm.prep()

        # Build the TT-approximation by the TT-CROSS method:
        Y = teneva.rand(bm.n, r=1, seed=seed)
        Y = teneva.cross(bm.get, Y, e=1.E-8, m=1.E+6, dr_max=1)
        Y = teneva.truncate(Y, e=1.E-8)
        r = teneva.erank(Y)

        # Generate full tensor and find its min/max values:
        Y_full = teneva.full(Y)
        y_min_real = np.min(Y_full)
        y_max_real = np.max(Y_full)

        # Find min/max values for TT-tensor by optima_tt:
        t = tpc()
        i_min, y_min, i_max, y_max = teneva.optima_tt(Y, k)
        t = tpc() - t

        # Calculate the errors:
        e_min = np.abs(y_min - y_min_real)
        e_max = np.abs(y_max - y_max_real)

        name = bm.name
        data[name] = {'t': t, 'r': r, 'e_min': e_min, 'e_max': e_max}

        text = ''
        text += name + ' ' * max(0, 18-len(name)) +  ' | '
        text += f'r: {r:-4.1f} | '
        text += f't: {t:-6.2f} | '
        text += f'e_min: {e_min:-7.1e} | '
        text += f'e_max: {e_max:-7.1e}'
        log(text)

    np.savez_compressed('result/data/function_small.npz', data=data,
        d=d, n=n, k=k)

    t_full = tpc() - t_full
    log(f'\n===> DONE | function_small | Time: {t_full:-10.3f}\n\n\n\n')

    show_function_small()


def calc_random_small(d_=[4,6], n_=[5,10], r_=[1,5], k=100, rep=100):
    t_full = tpc()

    log = Log('result/logs_calc/random_small.txt')
    log(f'---> CALC | random_small | k: {k:-4d}\n')

    data = {}

    for d in range(d_[0], d_[1]+1):
        data[d] = {}

        for r in range(r_[0], r_[1]+1):
            n = [np.random.choice(range(n_[0], n_[1]+1)) for _ in range(d)]
            t = 0.
            e_min = []
            e_max = []

            for _ in range(1, rep+1):
                # Create random TT-tensor of shape n and rank r:
                Y = teneva.rand(n, r)

                # Generate full tensor and find its min/max values:
                Y_full = teneva.full(Y)
                i_min_real = np.unravel_index(np.argmin(Y_full), n)
                i_max_real = np.unravel_index(np.argmax(Y_full), n)
                y_min_real = Y_full[i_min_real]
                y_max_real = Y_full[i_max_real]

                # Find min/max values for TT-tensor by optima_tt:
                t_cur = tpc()
                i_min, y_min, i_max, y_max = teneva.optima_tt(Y, k)
                t += tpc() - t_cur

                # Calculate the errors:
                e_min.append(abs(y_min - y_min_real))
                e_max.append(abs(y_max - y_max_real))

            t /= rep
            e_min = np.max(e_min)
            e_max = np.max(e_max)

            data[d][r] = {'t': t, 'e_min': e_min, 'e_max': e_max}

            text = ''
            text += f'd: {d:-4d} | '
            text += f'r: {r:-3d} | '
            text += f't: {t:-8.3f} * {rep:3d} | '
            text += f'e_min: {e_min:-7.1e} | '
            text += f'e_max: {e_max:-7.1e}'
            log(text)

    np.savez_compressed('result/data/random_small.npz', data=data,
        d_=d_, n_=n_, r_=r_, k=k, rep=rep)

    t_full = tpc() - t_full
    log(f'\n===> DONE | random_small | Time: {t_full:-10.3f}\n\n\n\n')

    show_random_small()


def calc_random_small_hist(d=6, n=16, r=3, k_=[1, 10, 25], rep=10000):
    t_full = tpc()

    log = Log('result/logs_calc/random_small_hist.txt')
    log(f'---> CALC | random_small_hist |\n')

    data = {}

    for k in k_:
        t = 0.
        e_min = []
        e_max = []

        for _ in range(1, rep+1):
            # Create random TT-tensor of shape n and rank r:
            Y = teneva.rand([n]*d, r)

            # Generate full tensor and find its min/max values:
            Y_full = teneva.full(Y)
            i_max_real = np.unravel_index(np.argmax(np.abs(Y_full)), [n]*d)
            y_max_real = Y_full[i_max_real]

            # Find max value for TT-tensor by optima_tt:
            t_cur = tpc()
            i_max, y_max = teneva.optima_tt_max(Y, k)
            t += tpc() - t_cur

            # Calculate the error:
            e_max.append(abs(y_max / y_max_real))

        t /= rep

        data[k] = {'t': t, 'e_max': e_max}

        text = ''
        text += f'k: {k:-4d} | '
        text += f't: {t:-8.3f} * {rep:3d} | '
        text += f'e_max: {np.mean(e_max):-7.1e}'
        log(text)

    np.savez_compressed('result/data/random_small_hist.npz', data=data,
        d=d, n=n, r=r, k_=k_, rep=rep)

    t_full = tpc() - t_full
    log(f'\n===> DONE | random_small_hist | Time: {t_full:-10.3f}\n\n\n\n')

    show_random_small_hist()


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/data')
    folder_ensure('result/plot')
    folder_ensure('result/logs_calc')
    folder_ensure('result/logs_show')

    mode = sys.argv[1] if len(sys.argv) > 1 else None

    if mode == 'fpe':
        calc_fpe()
    elif mode == 'function_big':
        calc_function_big()
    elif mode == 'function_small':
        calc_function_small()
    elif mode == 'random_small':
        calc_random_small()
    elif mode == 'random_small_hist':
        calc_random_small_hist()
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
