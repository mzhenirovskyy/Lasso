import numpy as np

from utils import timed, NotFittedError


class Lasso(object):
    """Linear Model trained with L1 prior as regularizer (aka the Lasso)
    The optimization objective for Lasso is::
        (1 / (2 * n_samples)) * ||Y - X*beta||^2_2 + alpha * ||beta||_1

    Attributes:
    alpha: float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        alpha = 0 is equivalent to an ordinary least square, solved
        by the "gradient descent".

    max_iter: int, optional
        The maximum number of iterations.

    tol: float, optional
        The tolerance for the optimization.
    """

    def __init__(self, alfa=1, max_iter=1000, tol=0.0001):
        """Inits Lasso."""

        self._alfa = alfa
        self._max_iter = max_iter
        self._tol = tol
        self._learning_rate = 0.001
        self._learning_rate_min = self._learning_rate / 1000
        self.beta = []
        self.X = []
        self.Y = []

    def _update_beta_number_j(self, number_j):
        """Does optimization along one parameter direction at the current
         point in each iteration during coordinate descent optimization.

         Args:
             number_j: int, number j (according to beta[j]).
         """

        beta_number_j_old = self.beta[number_j]
        max_iter_grad = 5000
        learning_rate = self._learning_rate
        residual = np.dot(self.X, self.beta) - self.Y
        cost = np.sum(residual ** 2)
        for k in range(max_iter_grad):
            self.beta[number_j] -= ((learning_rate / residual.size) *
                                    (np.dot(self.X.T, residual)[number_j]))
            if number_j > 0:
                self.beta[number_j] -= (learning_rate * self._alfa *
                                        np.sign(self.beta[number_j]))
            residual_new = np.dot(self.X, self.beta) - self.Y
            cost_new = np.sum(residual_new ** 2)
            if cost_new < cost:
                beta_number_j_old = self.beta[number_j]
                cost = cost_new
                residual = residual_new
            else:
                self.beta[number_j] = beta_number_j_old
                if learning_rate > self._learning_rate_min:
                    learning_rate /= 2
                else:
                    break

    def _coordinate_descent(self, n_beta):
        """Does coordinate descent optimization.

        Args:
             n_beta: int, number of parameters beta,
                     including beta0 - intercept.
        """

        residual = np.dot(self.X, self.beta) - self.Y
        cost = np.sum(residual ** 2)
        for k in range(self._max_iter):
            for j in range(n_beta):
                self._update_beta_number_j(j)
            residual_new = np.dot(self.X, self.beta) - self.Y
            cost_new = np.sum(residual_new ** 2)
            if abs(cost_new - cost) / cost_new < self._tol:
                print('Coordinate Descent has converged in %d iterations' % k)
                break
            else:
                cost = cost_new
        if k == self._max_iter - 1:
            print('Coordinate Descent has reached the maximum number'
                  ' of iterations %d ' % self._max_iter)

    def _gradient_descent(self):
        """Does gradient descent optimization.
        """

        residual = np.dot(self.X, self.beta) - self.Y
        cost = np.sum(residual ** 2)
        beta_old = np.copy(self.beta)
        for k in range(self._max_iter):
            self.beta -= ((self._learning_rate / residual.size) *
                          np.dot(self.X.T, residual))
            residual_new = np.dot(self.X, self.beta) - self.Y
            cost_new = np.sum(residual_new ** 2)
            if cost_new < cost:
                if abs(cost_new - cost) / cost_new < self._tol:
                    print('Gradient Descent has converged in %d iterations' % k)
                    break
                elif self._learning_rate < self._learning_rate_min:
                    print('Gradient Descent has reached the learning_rate_min')
                    break
                beta_old = np.copy(self.beta)
                cost = cost_new
                residual = residual_new
            else:
                self.beta = beta_old
                self._learning_rate /= 2
        if k == self._max_iter - 1:
            print('Gradient Descent has reached the maximum number'
                  ' of iterations %d ' % self._max_iter)
    @timed
    def fit(self, X, Y, beta0=None):
        """Fit model with coordinate descent in case alfa>0 and
           gradient descent if alfa=0.

        Args:
             X: ndarray, shape = (n_samples, n_features), data.
             Y: ndarray, shape = (n_samples, 1), target.
        Returns:
             self: trained (fitted) lasso model

        """

        self.X = np.atleast_2d(X)
        self.Y = np.atleast_1d(Y)
        n_samples, n_features = self.X.shape
        Z = np.ones(shape=(n_samples, 1))
        self.X = np.hstack((Z, self.X))
        n_beta = n_features + 1
        if not beta0:
            self.beta = np.zeros(shape=(n_beta,))
        else:
            self.beta = np.atleast_1d(beta0)
            self.beta = np.array(self.beta, dtype=np.float16)
        if self._alfa > 0:
            self._coordinate_descent(n_beta)
        else:
            print('alfa=0, so Gradient Descent will be used')
            self._gradient_descent()
        return self

    def predict(self, X):
        """Predict using the linear lasso model

        Args:
             X: ndarray, shape = (n_samples, n_features), samples.
        """

        if not len(self.beta):
            raise NotFittedError("This %(name)s instance is not fitted yet" %
                                 {'name': type(self).__name__})
        X = np.atleast_2d(X)
        n_sample, _ = X.shape
        Z = np.ones(shape=(n_sample, 1))
        X1 = np.hstack((Z, X))
        hypothesis = np.dot(X1, self.beta)
        return hypothesis
