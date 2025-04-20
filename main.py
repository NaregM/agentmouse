import numpy as np
import argparse

from train import train_agent
from config import *

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------

def main():
    """
    """
    parser = argparse.ArgumentParser(description="Train a DQN agent for Cat and Mouse")
    parser.add_argument("--role", choices=["mouse", "cat"], default="mouse",
                        help="Which agent to train: 'mouse' or 'cat'")
    args = parser.parse_args()

    # Select config based on role
    config = mouse_config if args.role == "mouse" else cat_config

    print(f"Starting training for: {args.role.upper()}")
    train_agent(config)
    



if __name__ == "__main__":
    main()
