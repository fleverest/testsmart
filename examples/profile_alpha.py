from testsmart.nnm import AlphaMart

import numpy as np

x = np.array([0] * 40000 + [1 / 2] * 30000 + [1] * 40000)
np.random.shuffle(x)

am = AlphaMart()
for xi in x:
    am.update([xi])
    if am.stopped:
        break

print(am.summaries.count)
