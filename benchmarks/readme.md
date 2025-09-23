# Benchmarks

This folder contains the different benchmarks we have run.

## Structure

Each benchmark run or family of benchmark runs has its own folder, with the python file needed to
run the benchmark and the putput .csv file. Each such folder should have a paragrap in the
[Benchmarks](#benchmarks) paragraph, detailing what the purpose of the benchamrk is, what
parameters were varied (and which weren't), and what the conclusions were.

## Benchmarks

### Initial benchmark

This mainly tests and demonstrates the usage of the benchmark module.

#### Model systems
The 3 and 6 dimensional Hartmann functions (`hart3` and `hart6`, respectively) are tested at an
expected random runtime of 1000. The noise levels are the default noise level (constant noise of
1% the span of the objective function, 0.038 for the 3 dimensional, 0.033 for the 6 dimensional),
in addition to 5 times lower and 5 times higher noise.

#### Suggestor
`n+1` and `3n` initial points are found with generalized golden ratio sampling. Then, an `Optimzer`
is used, with length scale bounds `[0.001, 1.0]` and noise level bounds `[0.0001, 1.0]`,
corresponding to a noise of 0.01 to 1, since the noise level bounds define the variance, not the
standard deviation.

#### Replicates
Each combination of the above settings is tested 50 times independently (with different seeds).

The total number of benchmarks runs is 600.