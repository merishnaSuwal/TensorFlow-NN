# TensorFlow-NN

Steps for regression2:
- Create variables
- Create placeholders
- define y model
- define loss function/ error
- define optimizer ( Gradient descent) to minimize loss function
- initialize global variable
- Run session ( Note: For large data set, use batches)

Steps for estimator API:
- Define a list of feature columns
- Create the estimator model
- Create a data input function
- Call train, evaluate, predict methods on the estimator object