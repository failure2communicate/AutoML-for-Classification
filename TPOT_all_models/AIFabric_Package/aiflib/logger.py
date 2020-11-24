from aiflib.config import Config
import logging

class Logger:
    def __init__(self, name):
        self.config = Config()
        self.logger_name = name
        self.logger = logging.getLogger(name)
        if self.config.is_info or self.config.is_verbose:
            self.logger.setLevel(logging.INFO)

    def info(self, msg):
        if self.config.is_info or self.config.is_verbose:
            self.logger.info(msg)
        if self.config.is_debug:
            self.logger.debug(msg)

    def verbose(self, msg):
        if self.config.is_verbose:
            self.logger.info(msg)
        if self.config.is_debug:
            self.logger.debug(msg)

    def debug(self, msg):
        self.logger.debug(msg)

class UiPathUsageException(Exception):
    pass