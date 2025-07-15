import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emulator.parameter_file_generator import ParameterFileGenerator 

def main(run_family_name = "default"):
    
    parameter_file_generator = ParameterFileGenerator(
        base_config_path="parameters.toml",
        run_family_name=run_family_name,
        inclusion_density_range=(0.5, 1.5),
        inclusion_speed_range=(1.0, 3.0),
        inclusion_radius_range=(0.03, 0.07),
        allow_inclusion_to_move=True,
        boundary_buffer=0.02,
        domain_size=0.25
       )
    
    parameter_file_generator.create_parameter_files(n_samples=100)
    output_directory = parameter_file_generator.output_dir

    # run the simulation for each parameter file
    for file in output_directory:
        sim = setup.build_simulator()
        sim.run()

    # get the data
    extractor = PPEDataExtractor(f"data/outputs/{run_family_name}")

if __name__ == "__main__":
    run_family_name = sys.argv[1]

    main(run_family_name)
