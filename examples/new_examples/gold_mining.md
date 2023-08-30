# Introduction to ProcessOptimizer

This file describes a series of Jupyter Notebooks (.ipynb files) that serves as an
introduction to what ProcessOptimizer is and how to use it.

This is meant to be a practical introduction. It won't explain the mathematical
underpinings of Bayesian optimization and Gaussian processes. It instead focuses on how
to use the package.

Each notebook assumes that you know the concepts introduced in previous notebooks, so it
can be a good idea to look at them in order.

## Main storyline

We follow the The Gold Mining Corporation's attempts to find as much gold as
possible in the plot of land it has gotten prospecting rights over. They need to decide
where to send their digging team, headed by Ms. Goldie Dixon, to dig, both to find gold,
but also to learn more about where there is gold. Since they
[start out](start_here.ipynb) not knowing anything about how much gold there is or
where, sothey use the ProcessOptimizer to suggest where to they should send Ms. Dixon
to dig.

At one point, [Ms. Dixon gets a new digging tool](categorical.ipynb) - A narrow drill.
Since it digs in a different way than the pickaxes she normally uses, the amount of gold
found is different depending on the tool she use. So now she has to choose the digging
tool as well as the the position of the dig.

We then [investigate the plots we use](plots.ipynb) in a bit more detail to see what we
can read from them. Here, we also introduce model uncertainty.

Two employees of The Gold Mining Corporation, Ms. Dixon and Mr. Drawson, disagree about
whether to find as much gold as possible
in suggested digs, or to try and get as much information as possible about where there
is gold in the plot. We look at how to get ProcessOptimizer to
[follow different approaches](explore_vs_exploit.ipynb), so that either Ms. Dixon or Mr.
Drawson get their way, or to strike a balance between the two approaches.

We also look at how to handle [getting several suggestions for digging sites at once](),
so that Ms. Dixon doesn't have to return to the main office after every dig, but can
just go to the next few digs, and only return to get new suggested digging sites once in
a while.

To see how [ProcessOptimizer handles constraints](contraints.ipynb), we give The Gold
Mining Corporation prospecting rights to a new plot of land, which isn't rectangular.

We discover silver in the original plot of land, and now The Metal Mining
Corporation determine where it is best to dig when the exchange rates keep changing.
They do this by finding [finding the Pareto front](pareto.ipynb); places where you can't
dig elsewhere and find both more silver and more gold. That way, they can always dig at
the optimal spot, regardless of the relative price of silver and gold.

We will also investigate how the distribution of gold affects the behaviour of
ProcessOptimizer - How good is it at finding small, concentrated gold vein; how does it
neahve when there is randomness in the amount of gold extracted?

## Deep analysis

These notebooks go into a lot more detail about how the ProcessOptimizer work, and how
to gain more knowledge about it. They are going in to a lot more theoretical detail than
the main storyline. They aren't required reading for using ProcessOptimizer, but if you
want a deeper understanding, they are recommended.

1. Kernel
2. Sampling from model
3. Scoring of model against ground truth
4. Length scale knowledge
5. Length scale prior
6. Noise level prior
7. Experimental budget
8. Change explore/exploit during optimisation
