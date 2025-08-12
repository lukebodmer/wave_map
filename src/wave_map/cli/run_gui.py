import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gui.user_interface import UserInterface 


ui = UserInterface(outputs_dir='data/outputs')
ui.show().servable()
