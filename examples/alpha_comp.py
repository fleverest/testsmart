import numpy as np
import seaborn as sns
import polars as pl

import matplotlib.pyplot as plt

from testsmart.nnm import AlphaMart, FixedBet, AGRAPA, ShrinkTrunc

np.random.seed(12345)


# For each model, run a single test and compare how each martingale evolves over time.

n_total = 1000
x = np.array([0] * 300 + [1 / 2] * 300 + [1] * 400)
np.random.shuffle(x)

tests = {
    "ALPHA + Shrink Trunc (eta0=0.75)": AlphaMart(
        n_total=n_total, estim=ShrinkTrunc(eta0=0.75)
    ),
    "ALPHA + Shrink Trunc (eta0=0.6)": AlphaMart(
        n_total=n_total, estim=ShrinkTrunc(eta0=0.6)
    ),
    "ALPHA + Shrink Trunc (eta0=0.55=True mean)": AlphaMart(
        n_total=n_total, estim=ShrinkTrunc(eta0=0.55)
    ),
    "ALPHA + Fixed Bet": AlphaMart(n_total=n_total, estim=FixedBet()),
    "ALPHA + aGRAPA": AlphaMart(n_total=n_total, estim=AGRAPA()),
}

count = []
test_name = []
e_proc = []
p_val = []
eta_estim = []

for name, test in tests.items():
    for xi in x:
        if test.stopped:
            break
        _ = test.update(np.array([xi]))
        count.append(test.summaries.count)
        test_name.append(name)
        e_proc.append(float(test.e_process[-1]))
        p_val.append(float(test.pval))
        eta_estim.append(float(test._eta_history[-1]))

df = pl.DataFrame(
    {
        "count": count,
        "test": test_name,
        "e-process": e_proc,
        "p-value": p_val,
        "eta": eta_estim,
    }
)

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="count", y="e-process", hue="test")
plt.axhline(20, ls="--")
plt.savefig("competition_e_process.png")

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="count", y="eta", hue="test")
plt.axhline(0.55, ls="--")
plt.savefig("competition_eta_estimates.png")

# Run same test with new sample 1000 times and compare stopping times for each model.

test_name = []
stop_time = []

for _ in range(1000):
    tests = {
        "ALPHA + Shrink Trunc (eta0=0.75)": AlphaMart(
            n_total=n_total, estim=ShrinkTrunc(eta0=0.75)
        ),
        "ALPHA + Shrink Trunc (eta0=0.6)": AlphaMart(
            n_total=n_total, estim=ShrinkTrunc(eta0=0.6)
        ),
        "ALPHA + Shrink Trunc (eta0=0.55=True mean)": AlphaMart(
            n_total=n_total, estim=ShrinkTrunc(eta0=0.55)
        ),
        "ALPHA + Fixed Bet": AlphaMart(n_total=n_total, estim=FixedBet()),
        "ALPHA + aGRAPA": AlphaMart(n_total=n_total, estim=AGRAPA()),
    }
    np.random.shuffle(x)
    for name, test in tests.items():
        for xi in x:
            if test.stopped:
                break
            _ = test.update(np.array([xi]))
        test_name.append(name)
        stop_time.append(test.summaries.count)

df = pl.DataFrame({"test": test_name, "stopping time": stop_time})

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="stopping time", y="test")
plt.savefig("competition_stopping_times.png")
