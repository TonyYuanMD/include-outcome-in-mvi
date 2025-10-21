"""Demonstration examples."""

from src.pipeline.simulation.simulator import SimulationStudy
from src.pipeline.simulation.missingness_patterns import MCARPattern
from src.pipeline.simulation.imputation_methods import MeanImputation

def example_basic():
    study = SimulationStudy(n=500, num_runs=1, seed=123)
    pattern = MCARPattern()
    method = MeanImputation()
    result = study.run_scenario(pattern, method)
    print(result)

if __name__ == "__main__":
    example_basic()