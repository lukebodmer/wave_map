import sys
from importlib import resources
from wave_map.simulator.simulation_setup import SimulationSetup
from wave_map.simulator.time_step_size_calculator import TimeStepSizeCalculator

def get_config_path(filename="parameters.toml"):
    try:
        # As pathlib.Path
        return resources.files("wave_map.config") / filename
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {filename} in wave_map.config")

def main(parameter_file=None, batch_name="single_run_simulations"):
    if parameter_file is None:
        parameter_file = get_config_path()

    calculator = TimeStepSizeCalculator(
        max_wave_speed=1.0,
        smallest_radii=0.004249573,
        polynomial_order=2
    )
    dt = calculator.calculate_cfl_dt()
    
    setup = SimulationSetup(
        dt=dt,
        config_path=str(parameter_file),
        batch_name=batch_name
    )
    sim = setup.build_simulator()
    sim.run()

if __name__ == "__main__":
    default_family = "default"
    
    if len(sys.argv) < 2:
        try:
            main(batch_name=default_family)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parameter_file = sys.argv[1]
        batch_name = sys.argv[2] if len(sys.argv) > 2 else default_family
        main(parameter_file, batch_name)
