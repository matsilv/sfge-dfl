The implementation of the method and its gradient computation is implemented respectively in the classes 
`BlackboxModule` and `Blackbox` of the `methods.blackbox` script. In the paper they only refer to minimization problem.
To make the method work without changing the optimization problem objective, we compute the perturbated input by adding 
the direction of the optimization as the gradient sign.