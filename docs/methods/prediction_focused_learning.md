This methods consists of training a machine learning model for maximum accuracy of the predictions. Once trained, the 
predictions of the model are then used to solve the downstream optimization problem.

We provide two (similar) ways of supervised learning:

- **Mean Squared Error (MSE) minimization**. The loss function is $\mathcal{L} = \sum_{i=1}^N || y^{(i)} -  \hat{y}^{(i)}||^2$,
where $y$ are the target values, $\hat{y}$ the predictions and $N$ is the training set size.
- **Negative log-likelihood minimization**. The loss function is 
$\mathcal{L} = -\sum_{i=1}^N \log \mathcal{N}(y;\hat{y}, \hat{\sigma})$, where $\mathcal{N}$ is a Normal distribution 
with mean $\hat{y}$ (the predictions of Machine Learning model) and standard deviation $\hat{\sigma}$, and $N$ is the training set size.

The first method is implemented in the `MSEModule` of the `methods.mse` script. The second one is implemented in the script 
`methods.prob_regressor`.