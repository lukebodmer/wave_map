import argparse
from wave_map.emulator.emulator_setup import EmulatorSetup


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for running the emulator testing."""
    parser = argparse.ArgumentParser(
        description="Run k-fold testing of the emulator for a given batch."
    )
    parser.add_argument(
        "batch_name",
        type=str,
        nargs="?",
        default="multi-cube-500",
        help="Name of the simulation batch to test (default: rotating_centered_ellipsoid_v2)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Setup the emulator and run validation
    emulator_setup = EmulatorSetup(batch_name=args.batch_name)
    emulator_setup.run_validation()

if __name__ == "__main__":
    main()
