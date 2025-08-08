import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emulator.parameter_file_generator import ParameterFileGenerator 
from emulator.emulation_setup import EmulationSetup 

def main(parameter_file):
    emulator = EmulationSetup(config_path=parameter_file)
    emulator.run()
    # test results
    # create PPE
    # test PPE 

if __name__ == "__main__":

    if len(sys.argv) < 1:
        print("Usage: python run_emulator.py <parameter_file>")
        sys.exit(1)

    run_name = sys.argv[1]

    main(run_name)
