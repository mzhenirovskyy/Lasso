import numpy as np
from sklearn.linear_model import Lasso as Sklearn_lasso
from linear_regretion import Lasso
from sklearn.metrics import r2_score

###############################################################################
# generate some data to play with
alpha = 0.1
np.random.seed(42)
n_samples, n_features = 50, 20
X = np.random.randn(n_samples, n_features)
coef = 3 * np.random.randn(n_features)
Y = np.dot(X, coef)

# add noise
Y += 0.01 * np.random.normal((n_samples,))

# Split data in train set and test set
n_samples = X.shape[0]
X_train, Y_train = X[:n_samples / 2], Y[:n_samples / 2]
X_test, Y_test = X[n_samples / 2:], Y[n_samples / 2:]

sklearn_lasso = Sklearn_lasso(alpha=alpha)
sklearn_predict = sklearn_lasso.fit(X_train, Y_train).predict(X_test)
sklearn_beta = np.hstack((sklearn_lasso.intercept_, sklearn_lasso.coef_))

lasso = Lasso(alfa=alpha)
predict = lasso.fit(X_train, Y_train).predict(X_test)
beta = lasso.beta

# Let estimate r2_score
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

sklearn_r2 = r2_score(sklearn_predict, Y_test)
print('sklearn_r2 = ', sklearn_r2)
r2 = r2_score(predict, Y_test)
print('r2 = ', r2)

r2_diff = abs(sklearn_r2 - r2) / sklearn_r2
print('r2_diff = ', r2_diff)

# Let see the L^2 vector norm of residual

norm2_residual = np.sum((predict - Y_test) ** 2)
print('norm2_residual = ', norm2_residual)

sklearn_norm2_residual = np.sum((sklearn_predict - Y_test) ** 2)
print('sklearn_norm2_residual = ', sklearn_norm2_residual)

norm2_diff = abs(sklearn_norm2_residual - norm2_residual) / sklearn_norm2_residual
print('norm2_diff = ', norm2_diff)

# ---------------------MY OUTPUT------------------------------------------
# Coordinate Descent has converged in 36 iterations
# fit() took 15 s to finish
# sklearn_r2 =  0.791335179159
# r2 =  0.761621887132
# r2_diff =  0.0375483016672
# norm2_residual =  607.715269732
# sklearn_norm2_residual =  573.334939003
# norm2_diff =  0.0599655251946