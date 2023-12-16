# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      de_opt
   Author :         zmfy
   DateTime :       2023/12/13 15:19
   Description :    
-------------------------------------------------
"""
import numpy as np
import math


class DEOptimizer:
    def __init__(self, n_dim, n_population=None, factor=0.5,
                 cross=0.8, max_iter=100, mutation_type="de/rand/1/bin"):
        """
        DE优化算法求解多元函数最小值（最大值）简单实现
        :param n_dim: 维度，即未知解数量
        :param n_population: 种群数量, 种群规模一般在[5*n_dim, 10*n_dim]之间, 默认7*n_dim
        :param factor: 缩放因子, 有效的缩放因子应该在0.4-1之间
        :param cross: 交叉概率
        :param max_iter: 最大迭代次数
        :param mutation_type: 变异操作类型，应为DE/rand/1/bin, DE/best/1/bin,
                            DE/rand-to-best/1/bin, DE/best/2/bin 或者 DE/rand/2/bin
        """
        self.factor = factor
        self.cross = cross
        self.max_iter = max_iter
        self.n_dim = n_dim
        self.npl = n_population if n_population is not None else int(7 * n_dim)
        self.mutation_type = mutation_type
        self.constraint_eq = []
        self.constraint_ueq = []

        self.X = None
        self.U, self.V = None, None
        self.lower, self.upper = None, None
        self.func = None
        self.mutation = self.mutation_factory(self.mutation_type)

        # 最优值
        self.best_x = None
        self.best_y = None
        self.current_best_idx = None    # 当前迭代的最优值所在的索引，即最优子代编号

        # 历史
        self.best_x_per_iter = []
        self.best_y_per_iter = []

    def init_values(self, func, lower, upper):
        self.func = func
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        # 初始化种群
        self.X = np.random.uniform(
            low=self.lower, high=self.upper,
            size=(self.npl, self.n_dim)
        )
        Y = func(self.X)
        self.current_best_idx = Y.argmin()

    def fit(self, func, upper_bound, lower_bound, constraint_eq=[], constraint_ueq=[]):
        """
        DE迭代入口函数
        :param func: 待优化函数
        :param upper_bound: 约束上界
        :param lower_bound: 约束下界
        :param constraint_eq: 一个函数句柄列表，每个函数代表一个等式约束
        :param constraint_ueq: 一个函数句柄列表，每个函数代表一个非等式约束
        :return: 返回最优(x, y)
        """
        self.init_values(func, lower_bound, upper_bound)
        self.constraint_eq, self.constraint_ueq = constraint_eq, constraint_ueq

        for n in range(self.max_iter):
            self.mutation()
            self.crossover()
            self.select(func)

            # 存储最优值
            Y = func(self.X)
            self.current_best_idx = Y.argmin()
            self.best_x_per_iter.append(self.X[self.current_best_idx, :].copy())
            self.best_y_per_iter.append(Y[self.current_best_idx].item())

        global_best_idx = np.array(self.best_y_per_iter).argmin()
        self.best_x = self.best_x_per_iter[global_best_idx]
        self.best_y = func(np.array([self.best_x]))

        return self.best_x, self.best_y

    def mutation_factory(self, mutation_type):
        if mutation_type.lower() == "de/rand/1/bin":
            return self._mutation_rand1
        elif mutation_type.lower() == "de/best/1/bin":
            return self._mutation_best1
        elif mutation_type.lower() == "de/rand-to-best/1/bin":
            return self._mutation_rand_to_best1
        elif mutation_type.lower() == 'de/best/2/bin':
            return self._mutation_best2
        elif mutation_type.lower() == 'de/rand/2/bin':
            return self._mutation_rand2
        else:
            raise ValueError(
                f"the param 'mutation_type' must be one of 'de/rand/1/bin', "
                f"'de/best/1/bin', 'de/rand-to-best/1/bin' or 'de/best/2/bin, "
                f"not '{mutation_type}'"
            )

    def _bound_constraint(self):
        # 边界约束, 将超出边界的值替换为边界内的随机值
        mask = np.random.uniform(
            low=self.lower, high=self.upper, size=(self.npl, self.n_dim)
        )
        self.V = np.where(self.V < self.lower, mask, self.V)
        self.V = np.where(self.V > self.upper, mask, self.V)

    def _penalty_eq(self):
        # 计算等式约束的罚函数
        penalty = 0
        for func in self.constraint_eq:
            penalty += func(self.X)
        return 1e5 * penalty

    def _penalty_ueq(self):
        # 计算非等式约束的罚函数
        penalty = 0
        for func in self.constraint_ueq:
            penalty += func(self.X)
        return 1e5 * penalty

    def _mutation_rand1(self):
        """
        DE/rand/1/bin变异操作， V[i]=X[r1] + F*(X[r2]-X[r3]), 要求种群数量>=4
        其中 i, r1, r2, r3 随机生成
        :return: 变异向量 V
        """
        X = self.X
        r1, r2, r3 = [], [], []
        # r1, r2, r3不应该相等
        while (np.array_equal(r1, r2) or np.array_equal(r1, r3) or
               np.array_equal(r2, r3)):
            idx = np.random.randint(0, self.npl, size=(self.npl, 3))
            r1, r2, r3 = idx[:, 0], idx[:, 1], idx[:, 2]

        # 执行突变
        self.V = X[r1, :] + self.factor * (X[r2, :] - X[r3, :])

        self._bound_constraint()

        return self.V

    def _mutation_best1(self):
        """
        DE/best/1/bin变异操作，
        X[r1, :] + f * (X[best,] - X[r1, :]) + f * (X[r2,] - X[r3, :])
        其中 i, r1, r2, r3 随机生成
        :return: 变异向量 V
        """
        X = self.X
        best = self.current_best_idx
        r1, r2 = [], []
        # r1, r2不应该相等
        while np.array_equal(r1, r2):
            idx = np.random.randint(0, self.npl, size=(self.npl, 2))
            r1, r2 = idx[:, 0], idx[:, 1]

        # 执行突变
        self.V = X[best, :] + self.factor * (X[r1, :] - X[r2, :])

        self._bound_constraint()

        return self.V

    def _mutation_rand_to_best1(self):
        """
        DE/best/1/bin变异操作，
        X[r1, :] + f * (X[best,] - X[r1, :]) + f * (X[r2,] - X[r3, :])
        其中 i, r1, r2, r3 随机生成
        :return: 变异向量 V
        """
        X = self.X
        best = self.current_best_idx
        r1, r2 = [], []
        # r1, r2不应该相等
        while np.array_equal(r1, r2):
            idx = np.random.randint(0, self.npl, size=(self.npl, 2))
            r1, r2 = idx[:, 0], idx[:, 1]

        # 执行突变
        self.V = X[best, :] + self.factor * (X[r1, :] - X[r2, :])

        self._bound_constraint()

        return self.V

    def _mutation_best2(self):
        """
        DE/best/2/bin变异操作，
        X[best, :] + f * (X[r1,] - X[r2, :]) + f * (X[r3,] - X[r4, :])
        其中 i, r1, r2, r3 随机生成
        :return: 变异向量 V
        """
        X = self.X
        best = self.current_best_idx
        r1, r2, r3, r4 = [], [], [], []
        # r1, r2不应该相等
        while (np.array_equal(r1, r2) or np.array_equal(r1, r3) or np.array_equal(r1, r4)
                or np.array_equal(r2, r3) or np.array_equal(r2, r4) or np.array_equal(r3, r4)):
            idx = np.random.randint(0, self.npl, size=(self.npl, 4))
            r1, r2, r3, r4 = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]

        self.V = X[best, :] + self.factor * (X[r1,] - X[r2, :]) + self.factor * (X[r3,] - X[r4, :])

        self._bound_constraint()
        return self.V

    def _mutation_rand2(self):
        """
        DE/rand/2/bin变异操作，
        X[r1, :] + f * (X[r2,] - X[r3, :]) + f * (X[r4,] - X[r5, :])
        其中 i, r1, r2, r3 随机生成
        :return: 变异向量 V
        """
        X = self.X
        r1, r2, r3, r4, r5 = [], [], [], [], []
        # r1, r2不应该相等
        while (np.array_equal(r1, r2) or np.array_equal(r1, r3) or np.array_equal(r1, r4) or np.array_equal(r1, r5)
               or np.array_equal(r2, r3) or np.array_equal(r2, r4) or np.array_equal(r2, r5) or np.array_equal(r3, r4)
                or np.array_equal(r3, r5) or np.array_equal(r4, r5)):
            idx = np.random.randint(0, self.npl, size=(self.npl, 5))
            r1, r2, r3, r4, r5 = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3], idx[:, 4]

        self.V = X[r1, :] + self.factor * (X[r2,] - X[r3, :]) + self.factor * (X[r4,] - X[r5, :])

        self._bound_constraint()
        return self.V

    def crossover(self):
        """
        交叉操作, 将目标变量 X 与变异变量 V 进行二项式交叉生成试验向量 U
        如果 随机数 < 交叉概率,则 U=V， 否则U=X
        :return: 试验向量 U
        """
        mask = np.random.rand(self.npl, self.n_dim) < self.cross
        self.U = np.where(mask, self.V, self.X)
        return self.U

    def select(self, fun):
        """
        选择操作，根据适应度选择是否保留 U
        :param fun: 目标函数
        :return: 目标向量 X
        """
        f_X = fun(self.X) + self._penalty_eq() + self._penalty_eq()
        f_U = fun(self.U) + self._penalty_eq() + self._penalty_eq()
        self.X = np.where((f_X < f_U).reshape(-1, 1), self.X, self.U)
        return self.X


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    def func_1(x):
        y = x * np.sin(x * math.pi * 10) + 2
        return y


    def func_2(x):
        y = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        return y


    opt = DEOptimizer(n_dim=2, mutation_type='de/best/1/bin')
    best_x, best_y = opt.fit(func_2, [1, 1], [-1, -1])
    print(f"最优点: x = {np.round(best_x, 6)}; 最优值: y = {np.round(best_y, 6)}")
    print('----------------')
    plt.plot(opt.best_y_per_iter, 'r-.')
    plt.ylabel("object fitness value")
    plt.xlabel("iterations")
    plt.show()
