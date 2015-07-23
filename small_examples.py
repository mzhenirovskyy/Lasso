import numpy as np
from sklearn.linear_model import Lasso as Sklearn_lasso
from linear_regretion import Lasso

# Let see some toy example from scikit-learn.org:
# http://scikit-learn.org/dev/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso
# and compare it with our lib.

print('-------------Example1: alfa=0.1---------------')
alfa = 0.1 # regularization coefficient
X_train = [[0, 0], [1, 1], [2, 2]]
Y_train = [0, 1, 2]
X_test = [[4, 5], [6, 7], [8, 9]]

lasso = Lasso(alfa=alfa)
predict = lasso.fit(X_train, Y_train).predict(X_test)

sklearn_lasso = Sklearn_lasso(alpha=alfa)
sklearn_predict = sklearn_lasso.fit(X_train, Y_train).predict(X_test)

print('sklearn_predict = ', sklearn_predict)
print('predict = ', predict)
sklearn_beta = np.hstack((sklearn_lasso.intercept_, sklearn_lasso.coef_))
print('sklearn_beta = ', sklearn_beta)
print('beta[:] = ', lasso.beta)

# Let see the L^2 vector norm of residual

norm2_residual = np.sum((sklearn_predict - predict) ** 2)
print('norm2_residual = ', norm2_residual)

# ---------------------MY OUTPUT------------------------------------------
# Coordinate Descent has converged in 22 iterations
# fit() took 5 s to finish
# sklearn_predict =  [ 3.55  5.25  6.95]
# predict =  [ 3.55018564  5.2501709   6.95015615]
# sklearn_beta =  [ 0.15  0.85  0.  ]
# beta[:] =  [  1.50012290e-01   8.49789778e-01   2.02848374e-04]
# norm2_residual =  8.80505759471e-08

###############################################################################

print('-------------Example2: alfa=1---------------')
alfa = 1 # regularization coefficient
X_train = [[0, 0], [1, 1], [2, 2]]
Y_train = [0, 1, 2]
X_test = [[4, 5], [6, 7], [8, 9]]

lasso = Lasso(alfa=alfa)
predict = lasso.fit(X_train, Y_train).predict(X_test)

sklearn_lasso = Sklearn_lasso(alpha=alfa)
sklearn_predict = sklearn_lasso.fit(X_train, Y_train).predict(X_test)

print('sklearn_predict = ', sklearn_predict)
print('predict = ', predict)
sklearn_beta = np.hstack((sklearn_lasso.intercept_, sklearn_lasso.coef_))
print('sklearn_beta = ', sklearn_beta)
print('beta[:] = ', lasso.beta)
norm2_residual = np.sum((sklearn_predict - predict) ** 2)
print('norm2_residual = ', norm2_residual)

# ---------------------MY OUTPUT------------------------------------------
# Coordinate Descent has converged in 1 iterations
# fit() took 0 s to finish
# sklearn_predict =  [ 1.  1.  1.]
# predict =  [ 0.99995483  0.99995483  0.99995483]
# sklearn_beta =  [ 1.  0.  0.]
# beta[:] =  [ 0.99995483  0.          0.        ]
# norm2_residual =  6.12189355988e-09

###############################################################################

print('-------------Example3: alfa=0.1---------------')
alfa = 0.1 # regularization coefficient
X_train = [[0, 0], [1 ,1], [0, 1], [1, 0], [2, 2]]
Y_train = [1, 6, 4, 3, 11]
X_test = [[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]]

lasso = Lasso(alfa=alfa)
predict = lasso.fit(X_train, Y_train).predict(X_test)

sklearn_lasso = Sklearn_lasso(alpha=alfa)
sklearn_predict = sklearn_lasso.fit(X_train, Y_train).predict(X_test)

print('sklearn_predict = ', sklearn_predict)
print('predict = ', predict)
sklearn_beta = np.hstack((sklearn_lasso.intercept_, sklearn_lasso.coef_))
print('sklearn_beta = ', sklearn_beta)
print('beta[:] = ', lasso.beta)

# Let see the L^2 vector norm of residual

norm2_residual = np.sum((sklearn_predict - predict) ** 2)
print('norm2_residual = ', norm2_residual)

# ---------------------MY OUTPUT------------------------------------------
# Coordinate Descent has converged in 20 iterations
# fit() took 2 s to finish
# sklearn_predict =  [  3.56520543   8.34785399  13.13050255]
# predict =  [  3.53190682   8.42406079  13.31621476]
# sklearn_beta =  [ 1.17388115  1.89141596  2.89123259]
# beta[:] =  [ 1.08582983  2.19584625  2.69630772]
# norm2_residual =  0.0414052968388




