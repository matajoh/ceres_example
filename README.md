# Ceres Example Project

This repository is intended to provide practical examples of basic Ceres usage, as
well as the infrastructure for a sample project which includes the library.
This is not intended as a replacement for the excellent and comprehensive
[Ceres Documentation](http://ceres-solver.org/), but rather as a pedogogical tool
and starting template for optimization projects.

## Getting Started

To build this project you will need a compiler and CMake 3.15 or later
(available [here](https://cmake.org/download/)). Once complete you can then
build the project via the following commands:

    cmake ..
    cmake --build . --config Release

This will build the `ceres_example` executable, which you can test
by simply typing:

    ./ceres_example

This should fit to a randomly generated dataset, resulting in
console output that looks like that shown below:

```
Numeric differentiation:
iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
   0  6.765364e+03    0.00e+00    3.00e+04   0.00e+00   0.00e+00  1.00e+04        0    5.01e-05    3.85e-04
   1  1.233953e+03    5.53e+03    4.02e+03   5.94e-01   8.18e-01  1.34e+04        1    4.35e-04    1.18e-03
   2  2.062309e+02    1.03e+03    5.50e+02   6.93e-01   8.33e-01  1.91e+04        1    1.34e-04    1.48e-03
   3  2.901531e+01    1.77e+02    7.86e+01   7.82e-01   8.60e-01  3.04e+04        1    1.25e-04    1.72e-03
   4  2.824528e+00    2.62e+01    1.17e+01   8.08e-01   9.05e-01  6.48e+04        1    1.62e-04    2.06e-03
   5  1.702931e-01    2.65e+00    1.50e+00   5.70e-01   9.65e-01  1.94e+05        1    1.37e-04    2.36e-03
   6  7.276579e-02    9.75e-02    7.72e-02   1.84e-01   9.97e-01  5.83e+05        1    1.14e-04    2.58e-03
   7  7.244271e-02    3.23e-04    3.18e-04   1.50e-02   1.00e+00  1.75e+06        1    1.26e-04    2.80e-03
trust_region_minimizer.cc:745 Terminating: Function tolerance reached. |cost_change|/cost: 2.005580e-07 <= 1.000000e-06
Ceres Solver Report: Iterations: 8, Initial cost: 6.765364e+03, Final cost: 7.244271e-02, Termination: CONVERGENCE
Initial: (h=0.1, k=0.3, a=0.9, b=1.2)
Final: (h=-0.262104, k=0.465688, a=4.24418, b=2.12967)
Target: (h=-0.3, k=0.5, a=4.3, b=2.1)
```

## Learning More

This repository shows how to use Ceres via the example of fitting an ellipse to a noisy arc of points, as seen
below:

![Ellipse fitting animation](docs/fit_ellipse_wide.gif)

In [`ceres_example.cpp`](ceres_example.cpp) you can view examples of:

1. Numeric cost functions
2. Autodiff cost functions
3. Analytic cost functions
4. Gradient checking
5. Solver callbacks
6. Solver customization

To learn more about Ceres and this example repository, please take a look at the
[tutorial slides](tutorial.md)
