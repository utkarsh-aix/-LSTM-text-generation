"""
Simple entry point for the Shakespeare LSTM project.

Usage:
    python main.py train
    python main.py generate
    python main.py experiment
"""
import sys

from src.train import main as train
from src.generate import main as generate
from src.experiment import main as experiment


def main():
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "train"

    if command == "train":
        train()
    elif command == "generate":
        generate()
    elif command == "experiment":
        experiment()
    else:
        print("Usage: python main.py [train|generate|experiment]")


if __name__ == "__main__":
    main()
