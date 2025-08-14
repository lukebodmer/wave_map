import sys
from importlib import resources
from wave_map.simulator.simulation_setup import SimulationSetup

def get_config_path(filename="parameters.toml"):
    try:
        # As pathlib.Path
        return resources.files("wave_map.config") / filename
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {filename} in wave_map.config")

def main(parameter_file=None, run_family_name="default"):
    if parameter_file is None:
        parameter_file = get_config_path()
    
    setup = SimulationSetup(
        config_path=str(parameter_file),
        run_family_name=run_family_name
    )
    sim = setup.build_simulator()
    sim.run()

if __name__ == "__main__":
    default_family = "default"
    
    if len(sys.argv) < 2:
        try:
            main(run_family_name=default_family)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parameter_file = sys.argv[1]
        run_family_name = sys.argv[2] if len(sys.argv) > 2 else default_family
        main(parameter_file, run_family_name)
