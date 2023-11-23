#!/usr/bin/env python3
"""
This file contains functions for logging
"""

class CustomLogger:
    def __init__(self, level="INFO"):
        self.levels = {"INFO": 1, "DEBUG": 2}
        self.set_level(level)

    def set_level(self, level):
        """Set the logging level."""
        self.level = self.levels.get(level.upper(), 1)

    def log(self, *args, level="INFO", **kwargs):
        """Custom log function, works like print but with log levels."""
        if self.levels[level.upper()] <= self.level:
            print(f"==={level.upper()}::", *args, **kwargs)

if __name__ == "__main__":
  # Usage
  logger = CustomLogger(level="DEBUG")

  # Example usage
  num_classes = 10
  unique_classes_test = set([1, 2, 3])

  # These will print like the standard print function, but with log level prefixes
  logger.log("Number of unique classes:", num_classes, level="INFO")
  logger.log("Number of unique classes test:", len(unique_classes_test), level="DEBUG")
