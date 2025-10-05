The method consists of training a machine learning model to minimize the regret by relying on the score function 
gradient estimation method. The gradient is computed as: 

$\bar{\eta} = \frac{1}{N} \sum_{i=1}^{N} \left( f(z^{*}(\hat{y}^{(i)}), y^{(i)}) - f(z^{*}(y^{(i)}), y^{(i)} \right) \nabla_{\theta} \log \mathcal{N}_{\theta} (\hat{y}^{(i)} | x)$,

with $\hat{y} \sim \mathcal{N}_{\theta} (y | x)$ where $x^*(\hat{q}^{(i)})$ is the optimal solution given the model 
predictions, $x^*(c^{(i)})$ is the optimal solution given the true model parameters, $c$ is the true cost vector, 
$\mathcal{N}_{\theta}$ is a gaussian distribution parametrized by $\theta$, $f$ is the input features vector and $N$ is 
the training set size. The method is implemented in the `SFGEModule` of the `methods.sfge` script.