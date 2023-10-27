from fpcross import EquationOUP
from fpcross import FPCross
import numpy as np
import os
import teneva
from time import perf_counter as tpc


class FpeOUP:
    def __init__(self, d=3, seed=42):
        self.seed = seed
        if d == 3:
            self.init_problem_3d()
        elif d == 4:
            self.init_problem_4d()
        elif d == 5:
            self.init_problem_5d()
        else:
            raise NotImplementedError()

        self.with_y_list = False

    def data_build(self, m, fpath=None, seed=None, tst=False):
        if seed is None:
            seed = self.seed

        t = tpc()
        if tst:
            I = teneva.sample_rand([self.n]*self.d_opt, m, seed)
        else:
            I = teneva.sample_lhs([self.n]*self.d_opt, m, seed)
        y = self.run_grid_many(I)
        t = tpc() - t

        if fpath:
            np.savez_compressed(fpath, data={'I': I, 'y': y, 't': t})

        return I, y, self.data_info(I, y, t, tst=tst, load=False)

    def data_info(self, I, y, t, tst=False, load=False):
        title1 = '\n------------> '
        title2 = 'Loaded dataset ' if load else 'Generated dataset '
        title3 = '(tst)\n' if tst else '(trn)\n'

        text = title1 + title2 + title3
        text += f'- Dimension : {I.shape[1]:-10d}\n'
        text += f'- Samples   : {I.shape[0]:-10.2e}\n'
        text += f'- Time full : {t:-10.2e}\n'
        text += f'- Time call : {t/len(I):-10.2e}\n'
        text += f'- Value min : {np.min(y):-10.4e}\n'
        text += f'- Value avg : {np.mean(y):-10.4e}\n'
        text += f'- Value max : {np.max(y):-10.4e}\n'
        text += '\n'
        return text

    def data_load(self, fpath, tst=False):
        data = np.load(fpath, allow_pickle=True).get('data').item()
        I, y, t = data['I'], data['y'], data['t']
        return I, y, self.data_info(I, y, t, tst=tst, load=True)

    def init_problem_3d(self):
        self.d = 3
        self.d_opt = self.d * self.d
        self.n = 11

        self.ind_to_poi = lambda i: i / 10

        rand = np.random.default_rng(self.seed)
        self.A_base = rand.random((self.d, self.d)) * 1.E-3

        self.A_opt = np.array([
            [0.8, 0.4, 0.1],
            [0.5, 0.9, 0.2],
            [0.3, 0.7, 0.6],
        ])

    def init_problem_4d(self):
        self.d = 4
        self.d_opt = self.d * self.d
        self.n = 11

        self.ind_to_poi = lambda i: i / 10

        rand = np.random.default_rng(self.seed)
        self.A_base = rand.random((self.d, self.d)) * 1.E-3

        self.A_opt = np.array([
            [0.8, 0.4, 0.1, 0.2],
            [0.5, 0.9, 0.2, 0.3],
            [0.3, 0.7, 0.6, 0.5],
            [0.4, 0.5, 0.4, 0.7],
        ])

    def init_problem_5d(self):
        self.d = 5
        self.d_opt = self.d * self.d
        self.n = 11

        self.ind_to_poi = lambda i: i / 10

        rand = np.random.default_rng(self.seed)
        self.A_base = rand.random((self.d, self.d)) * 1.E-3

        self.A_opt = np.array([
            [0.8, 0.4, 0.1, 0.2, 0.1],
            [0.5, 0.9, 0.2, 0.3, 0.2],
            [0.3, 0.7, 0.6, 0.5, 0.4],
            [0.4, 0.5, 0.4, 0.7, 0.3],
            [0.2, 0.1, 0.3, 0.2, 0.9],
        ])

    def loss(self, Y):
        dY = teneva.sub(self.Y_opt, Y)
        return teneva.norm(dY)

    def prep(self):
        self.Y_opt = self.run(self.A_opt)

    def run(self, A, with_y_list=False):
        eq = EquationOUP(d=self.d)
        eq.set_cross_opts(nswp=3, dr_min=0)
        eq.set_grid(n=15, a=-5., b=+5.)
        eq.set_grid_time(m=10, t=0.5)
        eq.set_coef_rhs(self.A_base + A)
        eq.init()
        eq.with_rs = False
        eq.with_rt = False

        fpc = FPCross(eq, with_hist=False, with_log=False,
            with_y_list=self.with_y_list)
        fpc.solve()

        if self.with_y_list:
            self.Y_list = fpc.Y_list

        return fpc.Y

    def run_grid(self, i):
        A = self.ind_to_poi(i).reshape(self.d, self.d)
        Y = self.run(A)
        return self.loss(Y)

    def run_grid_many(self, I):
        return np.array([self.run_grid(i) for i in I])
