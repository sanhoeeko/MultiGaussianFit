import numpy as np
import math as mh
import copy as cp
import matplotlib.pyplot as plt


def gaussian(x, mu, sigma2):
    return np.exp(-(x - mu) ** 2 / (2 * sigma2)) / mh.sqrt(2 * mh.pi * sigma2)


class MultiGaussianFit:
    def __init__(self, n_peaks: int, expected_peaks: int):
        """
        :param n_peaks: It may be more than the peaks you expect for better fitting
        :param expected_peaks: number of peaks you expect
        """
        self.n = n_peaks
        self.exceed_rate = expected_peaks / n_peaks
        self.mu = np.random.rand(n_peaks)
        self.sigma2 = np.ones(n_peaks) * 0.1
        self.a = np.ones(n_peaks) * 1
        self.sorted = False
        self.loss = None

    def fit(self, xs, ys, max_iter=1000):
        x = cp.deepcopy(xs)
        y = cp.deepcopy(ys)
        xmin = min(x)
        xmax = max(x)
        slope = xmax - xmin
        x = (x - xmin) / slope
        ymax = max(y)
        y /= ymax

        for i in range(max_iter):
            self._mgfit_01x01_iter(x, y)

        self.a *= self.exceed_rate
        self.mu = self.mu * slope + xmin
        self.sigma2 *= slope ** 2

    def __call__(self, x):
        ps = [self.a[j] * gaussian(x, self.mu[j], self.sigma2[j])
              for j in range(self.n)]
        return sum(ps)

    def _mgfit_01x01_iter(self, x, y):
        # calculate the ratio of each gaussian part at each position:
        # P_ij = gaussian(x_i | mu_j, sigma_j)
        ps = [self.a[j] * gaussian(x, self.mu[j], self.sigma2[j])
              for j in range(self.n)]
        P = np.vstack(ps)

        # Bayesian normalization:
        # Q_ij = P_ij / sum_j{P_ij}
        # now Q_ij represents the probability that sample x_i belongs to the distribution (mu_j, sigma_j)
        Q = P / np.sum(P, axis=0, keepdims=True)

        # update distributions
        # mu_j = sum_i{x_i y_i Q_ij} / sum_i{y_i Q_ij}
        weight_sum = Q @ y
        self.mu = Q @ (x * y) / weight_sum

        # sigma_j = sum_i{ (x_i-mu_j)^2 y_i Q_ij } / sum_i{y_i Q_ij}
        xi_muj = x.reshape((1, -1)) - self.mu.reshape((-1, 1))
        self.sigma2 = np.sum(Q * xi_muj ** 2 * y, axis=1) / weight_sum

        # a_j * sum_i{P_ij} = sum_i{y_i Q_ij} is the total count of each distribution
        self.a = weight_sum / np.sum(P, axis=1)

    def mse(self, x, y):
        y_pred = self(x)
        return np.sum((y_pred - y) ** 2) / len(y)

    def sort(self):
        # sort the peaks by mu, preparing for adding
        if not self.sorted:
            z = list(zip(self.mu, self.sigma2, self.a))
            z.sort(key=lambda x: x[0])
            self.mu, self.sigma2, self.a = tuple(map(np.array, zip(*z)))

    def __add__(self, o: 'MultiGaussianFit'):
        # merge many fittings for better precision
        mg = MultiGaussianFit(n_peaks=self.n, expected_peaks=1)
        mg.exceed_rate = self.exceed_rate
        mg.sorted = True
        # gamma = (1 / self.loss) / (1 / self.loss + 1 / o.loss)
        gamma = mh.exp(-self.loss) / (mh.exp(-self.loss) + mh.exp(-o.loss))
        mg.mu = gamma * self.mu + (1 - gamma) * o.mu
        mg.a = gamma * self.a + (1 - gamma) * o.a
        mg.sigma2 = gamma * self.sigma2 + (1 - gamma) * o.sigma2
        return mg

    def __radd__(self, o):
        # compatible with sum keysord
        if type(o) != MultiGaussianFit:
            return self
        else:
            return self + o

    def __lt__(self, o: 'MultiGaussianFit'):
        return False if self.loss < o.loss else True

    def __gt__(self, o: 'MultiGaussianFit'):
        return False if self.loss > o.loss else True


def average_mgfit(x: np.ndarray, y: np.ndarray, n_peaks: int, expected_peaks: int, n_times: int, max_iter=100):
    """
    :return: the 'average' GMM (in the sense of Taylor series) of the best 5 fittings.
    """
    mgs = [MultiGaussianFit(n_peaks, expected_peaks) for i in range(n_times)]

    def process(mg):
        mg.fit(x, y, max_iter)
        mg.sort()
        mg.loss = mg.mse(x, y)
        return mg

    mgs = list(map(process, mgs))
    mgs.sort(reverse=True)
    mgs = mgs[:5]
    if len(mgs) == 1:
        return mgs[0]

    res = 0
    for mg in mgs:
        res = res + mg
        res.loss = res.mse(x, y)

    return res


def best_mgfit(x: np.ndarray, y: np.ndarray, n_peaks: int, expected_peaks: int, n_times: int, max_iter=100):
    """
    :return: the best fitting among all <n_times> fittings. More stable and faster than the average one.
    """
    mgs = [MultiGaussianFit(n_peaks, expected_peaks) for i in range(n_times)]

    def process(mg):
        mg.fit(x, y, max_iter)
        mg.sort()
        mg.loss = mg.mse(x, y)
        return mg

    return max(map(process, mgs))


if __name__ == '__main__':
    xs = np.linspace(-2, 2, 1000)
    data = gaussian(xs, -1, 0.05) + 0.7 * gaussian(xs, 0,
                                                   0.05) + 0.5 * gaussian(xs, 0.7, 0.05)
    data += np.random.randn(1000) * 0.05
    data[data < 0] = 0
    # mg = average_mgfit(xs, data, n_peaks=5, expected_peaks=3, n_times=1000)
    mg = best_mgfit(xs, data, n_peaks=5, expected_peaks=3, n_times=1000)
    print(mg.mu, mg.sigma2, mg.a)
    print(mg.mse(xs, data))

    plt.plot(xs, data)
    plt.plot(xs, mg(xs))
    plt.show()
