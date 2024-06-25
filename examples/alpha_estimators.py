import numpy as np
import seaborn as sns
import polars as pl

from test_smart.nnm import AlphaMart, FixedBet, AGRAPA, ShrinkTrunc


n_total = 1000
n_sample = 500

x = np.random.uniform(0.1, 1, n_total)

alpha_st0 = AlphaMart(n_total=n_total)
alpha_st1 = AlphaMart(n_total=n_total, estim=ShrinkTrunc(eta0=.7))
alpha_fb = AlphaMart(n_total=n_total, estim=FixedBet())
alpha_agrapa = AlphaMart(n_total=n_total, estim=AGRAPA())

alpha_st0.update(x[:n_sample])
alpha_st1.update(x[:n_sample])
alpha_fb.update(x[:n_sample])
alpha_agrapa.update(x[:n_sample])

df = pl.DataFrame(
    {
        "index": np.tile(np.arange(1, n_sample + 1), 4),
        "e-process": np.array(
            alpha_agrapa.e_process
            + alpha_fb.e_process
            + alpha_st0.e_process
            + alpha_st1.e_process
        ),
        "p-value": np.array(
            alpha_agrapa.p_history
            + alpha_fb.p_history
            + alpha_st0.p_history
            + alpha_st1.p_history
        ),
        "estimator": (
            ["aGRAPA"] * n_sample
            + ["Fixed Bet"] * n_sample
            + ["Shrink Trunc default"] * n_sample
            + ["Shrink Trunc eta0=.7"] * n_sample
        ),
    }
)

plot1 = sns.lineplot(data=df, x="index", y="e-process", hue="estimator")
plot1.axhline(20, ls="--")
plot1.set_yscale("log")
plot1.get_figure().savefig("compare_estim_eproc.png")
plot1.clear()

plot2 = sns.lineplot(data=df, x="index", y="p-value", hue="estimator")
plot2.axhline(1 / 20, ls="--")
plot2.set_yscale("log")
plot2.get_figure().savefig("compare_estim_pval.png")
