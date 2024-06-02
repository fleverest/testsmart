# TEST Super MARTingales: `test_smart`

#### Working decription:
The `test_smart` python package may one day contain multiple utilities for
conducting certain sequential hypothesis tests using test (super)martingales.


#### Example: Wald's Sequential Probability Ratio Test (Exponential population)

Consider the simple-versus-simple hypothesis test for the parameter $\theta$ of
an exponential distribution:

```math
H_0: \theta = 1 \text{ versus }H_1: \theta = 2,\text{ where }\theta_1>\theta_0>0.
```

We can construct the corresponding SPRT as follows:

```python
import numpy as np
from test_smart.sprt import SPRT, ExponentialLogLikelihood, NormalLogLikelihood
exp_sprt = SPRT(
  alpha = 0.05,
  beta = 0.05,
  theta0 = 1,
  theta1 = 2,
  loglikelihood = ExponentialLogLikelihood()
)
norm_sprt = SPRT(
  alpha = 0.05,
  beta = 0.05,
  theta0 = 1,
  theta1 = 2,
  loglikelihood = NormalLogLikelihood(sigma = 1)
)
```

Then, given some observation(s) from the population we can update our test as
follows:

```python
exp_sprt.test(1.5)
# <Decision.CONTINUE: 'Continue testing'>
exp_sprt.test(np.array([2.00, 3.26, 2.46, 2.12, 4.88]))
# <Decision.REJECT: 'Reject the null hypothesis'>
exp_sprt.summary()
# {'null': 'theta = 1',
#  'alternative': 'theta = 2',
#  'loglikelihood': <Exponential(theta) log-Likelihood>,
#  'decision': 'Reject the null hypothesis',
#  'N': 6}
norm_sprt.test(1.5)
# <Decision.CONTINUE: 'Continue testing'>
norm_sprt.test(np.array([2.00, 3.26, 2.46, 2.12, 4.88]))
# <Decision.REJECT: 'Reject the null hypothesis'>
norm_sprt.summary()
# {'null': 'theta = 1',
#  'alternative': 'theta = 2',
#  'loglikelihood': <Normal(theta, 1)) log-Likelihood>,
#  'decision': 'Reject the null hypothesis',
#  'N': 6}
 ```