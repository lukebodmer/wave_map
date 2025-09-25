import argparse
import cProfile
import pstats
from pathlib import Path
from importlib import resources

from wave_map.batch_runner.batch_runner_setup import BatchRunnerSetup

# Constants
DEFAULT_CONFIG_FILENAME = "batch_parameters.toml"
PROFILE_OUTPUT_FILE = "profile_stats.txt"


def get_default_config(filename=DEFAULT_CONFIG_FILENAME) -> Path:
    """Return the path to the bundled default configuration file."""
    try:
        return resources.files("wave_map.config") / filename
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {filename} in wave_map.config")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for running a simulation batch."""
    parser = argparse.ArgumentParser(
        description="Run the simulation batch with a specified batch_parameter.toml file."
    )
    parser.add_argument(
        "batch_parameter_file",
        type=Path,
        nargs="?",
        default=get_default_config(),
        help=(
            "Path to the batch_parameter.toml file "
            f"(default: bundled {DEFAULT_CONFIG_FILENAME} in wave_map.config)"
        ),
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable cProfile performance profiling"
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=PROFILE_OUTPUT_FILE,
        help=f"File to save profiler stats (default: {PROFILE_OUTPUT_FILE})"
    )
    return parser.parse_args(argv)


def run_simulation_batch(parameter_file: Path):
    """Run a batch of simulations given a parameter file path."""
    if not parameter_file.exists():
        raise FileNotFoundError(f"Config file not found: {parameter_file}")

    print(f"Using config: {parameter_file}")
    batch_runner = BatchRunnerSetup(config_path=str(parameter_file))
    batch_runner.run()


def main(argv=None):
    args = parse_args(argv)

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

        run_simulation_batch(args.batch_parameter_file)

        profiler.disable()

        # Save stats to file
        with open(args.profile_output, "w") as f:
            stats = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
            stats.print_stats(50)  # top 50 entries

        print(f"Profile results saved to {args.profile_output}")
    else:
        run_simulation_batch(args.batch_parameter_file)


if __name__ == "__main__":
    main()
