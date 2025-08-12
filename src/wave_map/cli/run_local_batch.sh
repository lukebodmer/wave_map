#!/usr/bin/env bash

for config_file in inputs/parameter_configs/*.toml; do
    echo "Running simulation with config: $config_file"
    python run_simulation.py "$config_file"
    echo "Completed: $config_file"
    echo "---"
done
