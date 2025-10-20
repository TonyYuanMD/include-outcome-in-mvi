"""Demonstration examples."""

from simulator import SimulationStudy
from missingness_patterns import MCARPattern
from imputation_methods import MeanImputation

def example_basic():
    study = SimulationStudy(n=500, num_runs=1, seed=123)
    pattern = MCARPattern()
    method = MeanImputation()
    result = study.run_scenario(pattern, method)
    print(result)

if __name__ == "__main__":
    example_basic()