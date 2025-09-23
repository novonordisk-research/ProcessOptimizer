import csv
from itertools import product
from multiprocessing import Pool
from typing import Any

from ProcessOptimizer.benchmarking import BenchmarkInstance
from ProcessOptimizer.model_systems import get_model_system


def create_suggestor_definition(
    n_initial_points: int,
    length_scale_bounds: list[tuple[float, float]],
    noise_level_bounds: tuple[float, float],
) -> dict[str, Any]:
    return {
        "suggestor_name": "Sequential",
        "suggestors": [
            {"suggestor_name": "GoldenRatio", "suggestor_budget": n_initial_points},
            {
                "suggestor_name": "PO",
                "suggestor_budget": float("inf"),
                "acq_optimizer_kwargs": {
                    "length_scale_bounds": length_scale_bounds,
                    "noise_level_bounds": noise_level_bounds,
                },
            },
        ],
    }


def run_benchmark(benchmark: BenchmarkInstance):
    benchmark.run()
    # The benchmark instance needs to be returned, modifying it in the process
    # does not modify the original, so otherwise we can't get the results.
    return benchmark


if __name__ == "__main__":
    model_system_names = ["hart3", "hart6"]
    expected_random_runtime = 1000.0
    experimental_budget_per_dimension = 30
    noise_levels = [1.0, 5.0, 0.2]
    n_initial_points = ["n+1", "3n"]
    length_scale_bound = (0.001, 1.0)  # We need one per dimension
    noise_level_bounds = (0.0001, 1.0)
    n_replicates = 50
    validate = False
    seed = 0
    benchmarks: list[BenchmarkInstance] = []
    for model_system_name, noise_level, n_initial_point in product(
        model_system_names, noise_levels, n_initial_points
    ):
        model_system = get_model_system(model_system_name)
        n_dims = model_system.space.n_dims
        if n_initial_point == "n+1":
            n_initial_point = n_dims + 1
        elif n_initial_point == "3n":
            n_initial_point = 3 * n_dims
        else:
            raise ValueError(f"Unknown initial point configuration: {n_initial_point}")
        for _ in range(n_replicates):
            benchmarks.append(
                BenchmarkInstance(
                    model_system_name=model_system_name,
                    suggestor_definition=create_suggestor_definition(
                        n_initial_points=n_initial_point,
                        length_scale_bounds=[length_scale_bound] * n_dims,
                        noise_level_bounds=noise_level_bounds,
                    ),
                    experimental_budget=experimental_budget_per_dimension * n_dims,
                    expected_random_runtime=expected_random_runtime,
                    noise_level=noise_level,
                    validate=validate,
                    seed=seed,
                )
            )
            seed += 1

    with Pool(processes=8) as pool:
        benchmarks = pool.map(run_benchmark, benchmarks)
    # for benchmark in benchmarks:
    #    benchmark.run()

    first_dict = benchmarks[0].report
    with open(
        "benchmarks/initial_benchmark/benchmark_results.csv", "w", newline=""
    ) as csvfile:
        fieldnames = first_dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for benchmark in benchmarks:
            writer.writerow(benchmark.report)
