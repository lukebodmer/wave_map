import argparse

from wave_map.model_testers.model_tester import ModelTester


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for building the inverse model."""
    parser = argparse.ArgumentParser(
        description="Create an inverse model for a given simulation batch"
    )
    parser.add_argument(
        "batch_name",
        type=str,
        nargs="?",
        default="multi-cube-500",
        help="Name of the simulation batch to test (default: multi-cube-500)",
    )
    parser.add_argument(
        "model_name",
        type=str,
        nargs="?",
        default="None",
        help="Name of the inversion model to test (default: None)"
    )
    return parser.parse_args(argv)


def main(argv=None):

    # get simulation batch name and model name
    args = parse_args(argv)
    batch_name = args.batch_name
    model_name = args.model_name

    # create model
    model_tester = ModelTester(batch_name, model_name)
    model_tester.test_model()


if __name__ == "__main__":
    main()
