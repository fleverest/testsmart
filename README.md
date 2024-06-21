# TEST Super MARTingales: `test_smart`

#### Working decription:
The `test_smart` python package may one day contain multiple utilities for
conducting certain sequential hypothesis tests using test (super)martingales.


## Example: Wald's Sequential Probability Ratio Test


#### SPRT: Exponential scale parameter

Consider the simple-versus-simple hypothesis test for the scale parameter $\theta$ of
an exponential distribution:

```math
H_0: \theta = 1 \text{ versus }H_1: \theta = 2.
```

We can construct the corresponding SPRT as follows:

```python
import numpy as np
from test_smart.sprt import SPRT, ExponentialLogLikelihood
exp_sprt = SPRT(
  alpha = 0.05,
  beta = 0.05,
  theta0 = 1,
  theta1 = 2,
  loglikelihood = ExponentialLogLikelihood()
)
```

Then, given some observation(s) from the population we can update our test as
follows:

```python
exp_sprt.observe(0.5)
# <Decision.CONTINUE: 'Continue testing'>
exp_sprt.observe(1.5)
# <Decision.CONTINUE: 'Continue testing'>
exp_sprt.observe(1.7)
# <Decision.CONTINUE: 'Continue testing'>
exp_sprt.observe(1.9)
# <Decision.CONTINUE: 'Continue testing'>
exp_sprt.observe(1.0)
# <Decision.REJECT: 'Reject the null hypothesis'>
exp_sprt.summary()
# {'null': 'theta = 1',
#  'alternative': 'theta = 2',
#  'loglikelihood': <Exponential(theta) log-Likelihood>,
#  'decision': <Decision.REJECT: 'Reject the null hypothesis'>,
#  'N': 5}
```

#### SPRT: Normal location parameter

Consider the simple-versus-simple hypothesis test for the locationg parameter $\theta$
of a normal distribution:

```math
H_0: \theta = 1 \text{ versus }H_1: \theta = 2.
```

We can conduct the corresponding SPRT as follows:

```python
import numpy as np
from test_smart.sprt import SPRT, NormalLogLikelihood
norm_sprt = SPRT(
  alpha = 0.05,
  beta = 0.05,
  theta0 = 1,
  theta1 = 2,
  loglikelihood = NormalLogLikelihood(sigma = 1)
)
norm_sprt.observe(np.array([1.5, 0.0, 2.4, -1.0])) # Add a batch of observations
# <Decision.ACCEPT: 'Accept the null hypothesis'>
norm_sprt.summary()
# {'null': 'theta = 1',
#  'alternative': 'theta = 2',
#  'loglikelihood': <Normal(theta, 1) log-Likelihood>,
#  'decision': <Decision.ACCEPT: 'Accept the null hypothesis'>,
#  'N': 4}
```