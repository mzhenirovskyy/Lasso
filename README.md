LASSO regression: http://statweb.stanford.edu/~tibs/lasso.html

Motivation:
It is a test task.

Dependence:
The main module “linear_regretion.py” (based on requirement) has just numpy dependence. 

I have choose the “scikit-learn” (http://scikit-learn.org/stable/index.html) like reference implementation and compared my results with scikit-learn.
Therefore, to run “small_examples.py” and “big_example.py” scikit-learn need to be installed.

Boundary conditions:
I have 4 evenings (Mon - Thu) to be done. 

Realization:
Taking into account “Boundary conditions” the “coordinate descent” (https://en.wikipedia.org/wiki/Coordinate_descent) algorithm was selected, like simple in realization and not bad in performance.
If regularization parameter alpha equals zero the “gradient descent” applied.

Discussion:
1.	From the examples (small_examples.py and big_example.py) we can see that our implementation gives a very close results to scikit-learn.
2.	Of course, we lose in time. It’s understandable because scikit-learn uses low-level BLAS (via C interface).

