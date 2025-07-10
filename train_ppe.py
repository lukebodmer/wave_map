import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emulator.parameter_space_sampler import ParameterSpaceSampler 

def main(run_family_name = "default_family"):
    
    sampler = ParameterSpaceSampler('parameters.toml', run_family_name)

    # create the parameter files 
    output_directory = sampler.create_parameter_files()

    # run the simulation for each parameter file
    for file in output_directory:
        sim = setup.build_simulator()
        sim.run()

    # get the data
    extractor = PPEDataExtractor(f"data/outputs/{run_family_name}")

if __name__ == "__main__":
    run_family_name = sys.argv[1]

    main(run_name, run_family_name)
