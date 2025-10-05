Since the method is applicable when the predictions are linear in the cost vector, we extend the method to quadratic 
objective functions of the form $\frac{1}{2} z^TQz$ (with no linear term). SPO is implemented in the `methods.spo` 
script. In the same script, you can find the `SPO` and `QuadraticSPO` classes to compute the subgradient for 
respectively linear and quadratic objective functions. These classes extend `torch.autograd.Function`.