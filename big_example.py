import numpy as np
from sklearn.linear_model import Lasso as Sklearn_lasso
from linear_regretion import Lasso
from sklearn.metrics import r2_score, mean_squared_error

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

# Let see the rmse
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

rmse = mean_squared_error(predict, Y_test)
print('rmse = ', rmse)

sklearn_rmse = mean_squared_error(sklearn_predict, Y_test)
print('sklearn_rmse = ', sklearn_rmse)

# rmse for beta

rmse_beta = mean_squared_error(beta, sklearn_beta)
print('rmse_beta = ', rmse_beta)

# ---------------------MY OUTPUT------------------------------------------
# Coordinate Descent has converged in 36 iterations
# fit() took 15 s to finish
# sklearn_r2 =  0.791335179159
# r2 =  0.761621887132
# r2_diff =  0.0375483016672
# norm2_residual =  607.715269732
# sklearn_norm2_residual =  573.334939003
# norm2_diff =  0.0599655251946