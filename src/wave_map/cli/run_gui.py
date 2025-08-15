#!/usr/bin/env python3
import panel as pn
from wave_map.gui.user_interface import UserInterface

# Create the Panel app (this is what panel serve will use)
app = UserInterface(outputs_dir='data/simulation_batch_data').show()

# Required main() function for pyproject.toml script entry
def main():
    """Entry point for script usage"""
    pn.serve(app)
