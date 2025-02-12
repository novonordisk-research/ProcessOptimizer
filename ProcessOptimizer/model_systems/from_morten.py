# Name to use for the file we save the full results into
output_name_xlsx = 'SOME OUTPUT NAME.xlsx'

##% Import packages
import os
os.chdir(r"WHERE THE SCRIPT LIVES")
import numpy as np
from itertools import product
import time

import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.cm as mcolormap
# import matplotlib.lines as mlines
# from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

from ProcessOptimizer import Optimizer
# from ProcessOptimizer.plots import plot_objective, plot_objective_1d
from ProcessOptimizer.utils import expected_minimum
from ProcessOptimizer.model_systems import (
    hart3,
    hart6,
    peaks,
)


#%% Build all the model systems

# Build Peaks ModelSystem
peaks_model = peaks.create_peaks(noise=True)

# Build Hartmann 3D ModelSystem object
hart3_model = hart3.create_hart3(noise=True)

# Build Hartmann 6D ModelSystem object
hart6_model = hart6.create_hart6(noise=True)

# Gather model systems
# model_system_list = [
#     [peaks_model, "Peaks"],
#     [hart3_model, "Hartmann 3D"],
#     [hart6_model, "Hartmann 6D"],
# ]
model_system_list = [
    [hart6_model, "Hartmann 6D"],
]


#%% Settings for the full benchmark run

# Seeds for all random number generation. The length of this will determine how
# many times we run each model system.
seeds = range(0, 50)
# seeds = range(31,33) # For development

# The max of the star rating scale (for interpretability)
max_stars = 5 # Global for this benchmarking script

# Scaling strategy for n_initial_points
n_init_scale = ["n+1", "2n", "3n"]
# n_init_scale = ["n+1"]

# We will investigate two potential termination criteria:
    # 1: Stop when we think performance at the expected minimum will achieve 
    #    the goal 95 % of the time, trusting model and noise is accurate.
    # 2: When we think performance at the expected minimum will achieve the 
    #    goal 95 % of the time, we carry out one run at the x_EM, and stop if 
    #    we still think it will perform this well after this extra run.
# stop_criteria = ["trust", "validate"]
stop_criteria = ["validate"]

# Test with both the existing noise_level_bounds and strongly reduced bounds
# noise_bounds = [(1e-5, 1e5), (0.01, 0.5)]
noise_bounds = [(1e-5, 1e5)]

# The solution quality we are hunting in all benchmark systems, expressed as a
# fraction of the full score range that can be achieved inside the space
goals = [0.8]

# Define the fallback termination criterion for stopping when no solution is found
stop_fallbacks = [30] # Stop after n_dims * this number

# Define noise scales to test. We will use ConstantNoise with a noise scale of
# the full system range times the numbers below
# noise_scales = [0.01, 0.03, 0.05]
noise_scales = [0.03]

# The default optimizer normalizes Y

# Acquisition function used
acq_func_list = [
    "Naive EI", # Naive EI with no Xi "fudge" factor
    "Goal EI", # EI with Xi offset according to our goal
    # "Max uncertainty", # LCB with "inf" key-word (pick largest model uncertainty)
]

# Acqusition function hyperparameters
acq_func_opt_settings = [
    # "default",
    {"n_restarts_optimizer": 50},
]

# Strategy for replication
replication_list = [
    "no replicates",
    "initial_5",
]

dictionary_of_settings = {
    "seed": [0],
    "max stars": [max_stars],
    "stop criterion": stop_criteria,
    "optimization goal": goals,
    "termination fallback": stop_fallbacks,
    "noise level": noise_scales,
    "model system": model_system_list,
    "noise bounds": noise_bounds,
    "acqisition strategy": acq_func_list,
    "n_initial": n_init_scale,
    "replication strategy": replication_list,
    "acq func opt settings": acq_func_opt_settings,
}

# Wrap everything into a list of test settings to carry out
keys, values = zip(*dictionary_of_settings.items())
all_tests = [dict(zip(keys, p)) for p in product(*values)]

#%% Helper functions we will need



# Helper function that calculates the initial number of runs automatically
def initial_run_number(model_system):
    # At least 5 when the systems are small, n_dims+1 otherwise
    n_init = max(5, model_system.space.n_dims+1)
    return n_init

# Helper function that converts a score to a star rating number
def star_score(Y, max_stars, model_system):
    # Get the scale of the system
    y_scale = model_system.true_max - model_system.true_min
    # Map Y into the scale 0 to max_stars
    score = (model_system.true_max - Y) / y_scale * max_stars
    # Flip the sign of score, since the algoritm minimizes
    score = -score
    return score


# Run the benchmark using a set of settings
def benchmark_tester(dictionary_of_settings):
    model = dictionary_of_settings.get("model system")[0]
    seed = dictionary_of_settings.get("seed")
    # Setup noise generator with seed and noise size
    model.noise_model.set_seed(seed)
    noise_level = dictionary_of_settings.get("noise level")
    noise_size = (model.true_max - model.true_min) * noise_level
    model.noise_model.noise_size = noise_size
    # Get our optimization goal
    y_range = model.true_max - model.true_min
    real_goal = model.true_min + y_range*(1-dictionary_of_settings.get("optimization goal"))
    goal = -dictionary_of_settings.get("optimization goal")*max_stars
    
    # Get initial and max number of runs
    n_init_strat = dictionary_of_settings.get("n_initial")
    if n_init_strat == "n+1":
        n_init = model.space.n_dims+1
    elif n_init_strat == "2n":
        n_init = model.space.n_dims*2
    elif n_init_strat == "3n":
        n_init = model.space.n_dims*3
        
    n_max = dictionary_of_settings.get("termination fallback")*model.space.n_dims
    
    # Get experimental stop criterion
    stop_crit = dictionary_of_settings.get("stop criterion")

    # Get acquisition function according to strategy
    acq_strategy = dictionary_of_settings.get("acqisition strategy")
    if acq_strategy in ["Naive EI", "Goal EI"]:
        acq_func = "EI"
    elif acq_strategy == "Max uncertainty":
        acq_func = "LCB"
      
    # Get replication strategy
    repl_strategy = dictionary_of_settings.get("replication strategy")
    
    # Get acquisition function settings, if relevant
    acq_opt_kwargs = dictionary_of_settings.get("acq func opt settings")
      
    # Build optimizer
    opt = Optimizer(
        dimensions=model.space,
        lhs=False,
        acq_func=acq_func,
        n_initial_points=n_init,
        random_state=seed,
    )
    # Set acquisition function settings. If we're using Naive EI or maximum
    # uncertainty, we won't need to ever update them.
    dfi = 0 # also known as xi, represents desired further improvement
    kappa = "inf"
    opt.acq_func_kwargs={"xi": dfi, "kappa": kappa}
    
    # Set acqusition function optimizer settings, if relevant
    if acq_opt_kwargs != "default":
        opt.acq_optimizer_kwargs = acq_opt_kwargs
    
    # Apply the desired noise_level_bounds
    noise_bounds = dictionary_of_settings.get("noise bounds")
    opt.base_estimator_.noise_level_bounds = noise_bounds  
    
    # Get initial points using golden ratio sampling
    next_x = grs(model, n_init, seed)
    
    # Get results and feed to model
    for x in next_x:
        y = star_score(model.get_score(x), max_stars, model)
        res = opt.tell(x, y)
    
    # If we are using the initial replications strategy, run the specified number
    # of extra center points at the beginning
    if repl_strategy == "initial_5":
        x = [0.5] * model.space.n_dims
        for i in range(5):
            y = star_score(model.get_score(x), max_stars, model)
            res = opt.tell(x, y)
    
    validated = False
    success = False
    # Continue iterative optimization up until our allowed maximum runs
    while len(opt.yi) < n_max:
        # After a number of simulations it becomes apparent that Max uncertainty 
        # performs horrifically in complicated systems, so don't spend time on
        # this strategy any more
        if acq_strategy == "Max uncertainty":
            success = False
            break
        
        # Gather information on the expected minimum at present including the
        # modelled measurement noise
        opt.add_observational_noise()
        res = opt.get_result()
        result_location, [result_value, result_std] = expected_minimum(res, return_std=True)
        opt.remove_observational_noise()
        # Stop if we think we are done
        if result_value + 2*result_std < goal:
            if stop_crit == "trust":
                real_result = model.score(result_location)
                # We succeed if our chosen settings meet the goal
                success = real_result + 2*noise_size < real_goal
                break
            elif stop_crit == "validate":
                if validated:
                    # We arrive here when we have just done an experiment in the
                    # expected minimum and this did not change that the model
                    # thinks the result will be acceptable
                    real_result = model.score(result_location)
                    # We succeed if our chosen settings meet the goal
                    success = real_result + 2*noise_size < real_goal
                    break
                else:
                    # We arrive here the first time the model thinks our
                    # solution is good enough. We do an experiment at the expected
                    # minimum and go back to the beginning to ensure we still
                    # think the solution is good enough after checking
                    x = result_location
                    y = star_score(model.get_score(x), max_stars, model)
                    res = opt.tell(x, y)
                    validated = True
        else:
            if acq_strategy == "Goal EI":
                # Calculate desired further improvement at this stage
                dfi = max([result_value + 2*result_std - goal, np.abs(0.01*goal)])
            
            # Update the optimizer
            opt.acq_func_kwargs={"xi": dfi, "kappa": kappa}            
            opt.update_next()
            # Get next experiment
            x = opt.ask(1)
            # Get result and feed to model
            y = star_score(model.get_score(x), max_stars, model)
            res = opt.tell(x, y)
            # If we make it down here, we will reset the validated flag
            validated = False
    
    return opt, res, success

#%% Manual override for when some tests have already been run

override = False

skip_to = 0 # inclusive
if override:
    print("Skipping first {} tests".format(skip_to))

#%% Tell the user how hard each problem is

for i, test in enumerate(all_tests):
    # If some tests have already been run, don't repeat them
    if override and i < skip_to:
        continue
    # Calculate how long it would take random experiments to find the solution
    # on average
    rERT = random_ERT(test)
    # Store this information in the test
    test["Random ERT"] = rERT
    print("System: {}, Noise: {}".format(test.get("model system")[1], test.get("noise level")))
    print("The random ERT of test {} is {}".format(i, rERT))
    print("===========")

#%% Carry out all the tests

# Change to our output directory
os.chdir("./SOMEOUTPUT FOLDER THAT EXISTS")

overall_results = []
for i, test in enumerate(all_tests):
    # If some tests have already been run, don't repeat them
    if override and i < skip_to:
        continue
    
    print(time.strftime("%D") + " " + time.strftime("%H:%M:%S"))
    print("Running test {} of {}, settings:".format(i+1, len(all_tests)))
    print(test, end="\n")
    
    single_opt = []
    single_res = []
    single_success = []
    single_iterations = []
    single_model_noise = []
    for seed in seeds:
        print("\r... Seed: {}".format(seed), end="", flush=True)
        test["seed"] = seed
        opt, res, success = benchmark_tester(test)
        # Store the individual optimizers, result objects, successes and iterations
        single_opt.append(opt)
        single_res.append(res)
        single_success.append(success)
        single_iterations.append(len(opt.yi))
        # Also extract the full noise in the model
        opt_noise = opt.copy()
        opt_noise.add_observational_noise()
        res = opt_noise.get_result()
        _, [_, result_std] = expected_minimum(res, return_std=True)
        single_model_noise.append(result_std)
    
    # Store information about system name, real noise and modelled noise
    test["model name"] = test.get("model system")[1]
    test["real noise"] = max_stars * test.get("noise level")
    test["average noise"] = np.mean(single_model_noise)
    test["std noise"] = np.std(single_model_noise, ddof=1)
    test["model noises"] = single_model_noise
    # Store the optimizers, models, iteration and success lists in the test
    test["optimizers"] = single_opt
    test["models"] = single_res
    test["iterations"] = single_iterations
    test["mean iterations"] = np.mean(single_iterations)
    test["std iterations"] = np.std(single_iterations, ddof=1)
    test["success"] = single_success
    test["total success"] = sum(single_success)
    # Summarize performance
    if sum(single_success) > 0:
        test["ERT"] = sum(single_iterations)/sum(single_success)
    else:
        test["ERT"] = "Inf"
    print("\n")
    print("Test complete, ERT was: {}".format(test["ERT"]))
    print("-----------------------------")   
    
    
    # Save the outcome of this test as a back-up if the simulation is not run
    # to completion
    df = pd.DataFrame([test])
    df.to_excel("BO ERT test {}.xlsx".format(i))
    
    overall_results.append(test)


#%% Save combined results using pandas

df2 = pd.DataFrame(overall_results)
df2.info()
df2.to_excel(output_name_xlsx)


#%% Display that we are done
print("\n")
print("Full simulation script completed at " + time.strftime("%D") + " " + time.strftime("%H:%M:%S"))
print("\n")

