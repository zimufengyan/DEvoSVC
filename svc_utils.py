# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      svc_utils
   Author :         zmfy
   DateTime :       2023/12/14 13:34
   Description :    基于DE优化的SVC简单实现
-------------------------------------------------
"""
import numpy as np
from sklearn.svm import SVC
from scipy.spatial import distance

from de_opt import DEOptimizer


class DEvoSVC:
    def __init__(self, C=1.0, gamma='scale', **args):
        """
        一个简单的基于DE差分进化优化的高斯核SVC二分类器
        :param C: float，惩罚参数
        :param gamma: 核函数参数, 必须为 'scale', 'auto', 或者一个浮点数
        :param args: 额外参数，DE优化器用
        """
        if gamma != "scale" and gamma != "auto" and not (isinstance(self.gamma, float) or isinstance(self.gamma, int)):
            raise ValueError(f"the parameter 'gamma' must be 'scale', 'auto', or a float, not {gamma}")

        self.C = C
        self.gamma_type = gamma
        self.gamma = None
        self.args = args

        self.X, self.y = None, None
        self.alphas = None
        self.bias = None    # 偏置
        self.is_sv = None
        self.margin_sv_label = None
        self.support_vectors = None
        self.support_vector_alphas = None
        self.support_vector_labels = None

        self.optimizer = None

    def init_optimizer(self, n_samples):
        n_population = self.args.get('n_population', None)
        mutation_type = self.args.get('mutation_type', "de/rand/1/bin")
        factor = self.args.get('factor', 0.6)
        cross = self.args.get('cross', 0.9)
        max_iters = self.args.get('max_iters', 500)
        optimizer = DEOptimizer(
            n_dim=n_samples,
            n_population=n_population,
            factor=factor,
            cross=cross,
            max_iter=max_iters,
            mutation_type=mutation_type
        )
        return optimizer

    def rbf(self, x1, x2):
        return np.exp(-self.gamma * distance.cdist(x1, x2, 'sqeuclidean'))

    def fit(self, X: np.ndarray, y: np.ndarray, eval_train=True):
        X, y = np.array(X), np.array(y).reshape(-1, 1)
        self.X, self.y = X, y
        n_samples, n_features = X.shape
        if self.gamma_type == "scale":
            self.gamma = 1 / (n_samples * np.var(X))
        elif self.gamma_type == 'auto':
            self.gamma = 1 / n_samples
        else:
            self.gamma = float(self.gamma_type)

        # if labels given in {0,1} change it to {-1,1}
        old_y = y.copy()
        if set(np.unique(y)) == {0, 1}:
            y[y == 0] = -1

        # compute the kernel over all possible pairs of (x, x') in the data
        K = self.rbf(X, X)

        # define optimization problem
        # x (i.e., alphas), including n_samples vars, shape (n_population, n_samples)
        def func(x):
            q = np.ones((1, n_samples))  # (1, n)
            # batch mul
            a = np.expand_dims(x, axis=1)  # (n_population, 1, n_samples)
            b = np.expand_dims(x, axis=2)  # (n_population, n_samples, 1)
            tmp = (y @ y.T * K)
            # einsum('ijk,kp->ijp, A, B):
            # 爱因斯坦求和约定函数, 第一个参数为求和约定字符串，其中ijk表示A的维度，kp表示B的维度，ijp表示结果维度
            # 利用该函数计算张量与矩阵的乘积
            tmp = np.einsum('ijk,kp->ijp', a, tmp, optimize=True)  # (n_population, 1, n_sample)
            former = np.einsum('ijk,ikp->ijp', tmp, b, optimize=True)  # (n_population, 1, 1)
            latter = np.einsum('ijk,kp->ijp', a, q.T, optimize=True)
            return 0.5 * np.squeeze(former, axis=2) - np.squeeze(latter, axis=2)

        upper, lower = [self.C] * n_samples, [0] * n_samples
        constraint_eq = [
            # (1, n_samples) * (n_population, n_samples, 1) -> (n_population, 1, 1)
            lambda x: np.squeeze(
                np.einsum('jk,ikp->ijp', y.T, np.expand_dims(x, axis=2), optimize=True),
                axis=2
            )  # y.T @ x = 0
        ]

        # perform DE optimization
        self.optimizer = self.init_optimizer(n_samples)
        best_alphas, _ = self.optimizer.fit(func, upper, lower, constraint_eq)
        self.alphas = best_alphas.reshape(-1, 1)

        # Select the support vectors
        eps = 1e-6
        is_sv = ((self.alphas - eps > 0) & (self.alphas <= self.C - eps)).squeeze()
        self.support_vectors = X[is_sv]
        self.support_vector_labels = y[is_sv]
        self.support_vector_alphas = self.alphas[is_sv]

        # compute bias
        # an index of some margin support vector
        margin_sv = np.argmax((0 < self.alphas - eps) & (self.alphas < self.C - eps))
        margin_sv_label = y[margin_sv]
        margin_sv_x = X[margin_sv, np.newaxis]
        self.bias = margin_sv_label - np.sum(
            self.support_vector_alphas * self.support_vector_labels * self.rbf(X, margin_sv_x),
            axis=0)

        if eval_train:
            print(f"Finished training with accuracy {self.evaluate(X, old_y)}")

    def predict(self, X):
        X = np.array(X)
        score = np.sum(
            self.support_vector_alphas * self.support_vector_labels * self.rbf(self.support_vectors, X),
            axis=0) + self.bias     # 不加偏置
        pred = np.sign(score).reshape(-1, 1)
        pred[pred == -1] = 0
        return pred

    def evaluate(self, X, y):
        X, y = np.array(X), np.array(y).reshape(-1, 1)
        outputs = self.predict(X)
        accuracy = np.sum(outputs == y) / len(y)
        return round(accuracy, 6)
