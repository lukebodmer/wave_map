import logging
from pathlib import Path

class Logger:
    def __init__(self, log_path: Path, name: str = "emulatorlog", level=logging.INFO):
        self.log_path = log_path
        self.name = name
        self.level = level
        self._logger = logging.getLogger(self.name)
        self._configure()

    def _configure(self):
        # Avoid duplicate handlers if Logger is re-instantiated
        if self._logger.hasHandlers():
            self._logger.handlers.clear()

        self._logger.setLevel(self.level)

        formatter = logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        terminal_formatter = logging.Formatter("%(message)s")

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(terminal_formatter)
        self._logger.addHandler(ch)

        # File handler
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(self.log_path, mode="w")
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def debug(self, msg: str):
        self._logger.debug(msg)

    def get(self):
        return self._logger  # if you need to pass it into 3rd party APIs
