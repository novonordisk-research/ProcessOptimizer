# Gold mining examples (Walkthrough)

The gold mining examples are a series of Jupyter notebooks that introduce the basic use
of the `ProcessOptimizer` through a simple example. The example follows an attempt to find
the places in a plot of land where there is most gold.

In [the first part](01_start_here.ipynb), we are introduced to plot and to our digger, Ms.
Goldie Dixon. We learn how to define the parameter space, start a `ProcessOptimizer`, ask
it where to dig, and telling it how much gold we found. We also discuss whether to stop
early, or keep going in the hopes of finding an even better spot.

We then [learn about batch vs. single experiments](02_batch_vs_single.ipynb). This covers
whether we tell the `ProcessOptimizer` about all of the experiments before asking for a
new recommendation, or whether we ask for a list of new experiments to perform, do them,
and only then tell `ProcessOptimizer` about the all of the results.

In the third part, we [look at noisy data](03_noise.ipynb) - Up until now, the amount of
gold found has simply been a function of the dig position. Now, we introduce some
uncertainty as to how much is dug up, so if you dig at the same place twice, you won't
find the same amount.

The we look at whether to use `ProcessOptimizer` to be sure that we find a lot of gold at
each dig, or whether to risk not finding a lot, because it also means that we might find
a lot more. This is calle [the explore vs. exploit balance](04_explore_vs_exploit.ipynb).

In the fifth part, we introduce [small areas with a lot of gold](05_narrow_wells.ipynb),
and see how `ProcessOptimizer` handles that.

We also look at the different
ways we can [plot the result of `ProcessOptimizer`](06_plots.ipynb).

But what if you have two disparate objectives, like cost and quality? We investigate how
to handle that when we
[look at multiobjective optimization](07_multibjective.ipynb).