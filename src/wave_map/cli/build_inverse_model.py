import argparse

from wave_map.data_processor.data_processor import DataProcessor
from wave_map.inverse_models.parallel_partial_emulator import ParallelPartialEmulator
#from wave_map.inverse_models.neural_networks import NeuralNetwork


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
        "model_type",
        type=str,
        nargs="?",
        default="neural_network",
        help="Type of inverse model to create (options: neural_network(default), parallel_partial_emulator)",
    )
    return parser.parse_args(argv)


def main(argv=None):

    # get model type and simulation batch name
    args = parse_args(argv)
    model_type = args.model_type
    batch_name = args.batch_name

    # process data
    data_processor = DataProcessor(batch_name=batch_name)
    data_processor.save_processed_training_files()

    # create model
    if model_type == "neural_network":
        #model = NeuralNetwork(batch_name)
        model = ParallelPartialEmulator(batch_name)
    elif model_type == "parallel_partial_emulator":
        model = ParallelPartialEmulator(batch_name)

    # train model
    model.train(test_set_size=0.1)


if __name__ == "__main__":
    main()
