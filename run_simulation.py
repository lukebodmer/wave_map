import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from wave_simulator.simulation_setup import SimulationSetup
#from wave_simulator.parameter_space_sampler import ParameterSpaceSampler 
import cProfile

def main(parameter_file, run_family_name = "default_family"):
    
    #profiler = cProfile.Profile()
    #profiler.enable()
    #sampler = ParameterSpaceSampler('parameters.toml').create_parameter_files()
    #extractor = PPEDataExtractor("./outputs")
    setup = SimulationSetup(config_path=parameter_file,
                            run_family_name=run_family_name)
    #X, Y = extractor.extract()
    #print("Input shape (X):", X.shape)
    #print("Output shape (Y):", Y.shape)

    # Optional: Save
    #np.save("ppe_inputs.npy", X)
    #np.save("ppe_outputs_pressure.npy", Y)
    sim = setup.build_simulator()
    sim.run()
    #profiler.disable()
    #profiler.dump_stats('profile_results.prof')  # Save for analysis

if __name__ == "__main__":
    import sys

    # run_name = "my_sim"
    # run_family_name = "default"
    default_family = "default"

    if len(sys.argv) < 2:
        print("Usage: python script.py <parameter_file> [<run_family_name>]")
        sys.exit(1)

    run_name = sys.argv[1]
    run_family_name = sys.argv[2] if len(sys.argv) > 2 else default_family

    main(run_name, run_family_name)
