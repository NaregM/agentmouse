import numpy as np
import matplotlib.pyplot as plt
import argparse

from helpers import running_mean
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
    losses = train_agent(config)
    
    plt.figure(figsize=(10,7))
    plt.plot(running_mean(np.array(losses)))
    plt.xlabel("Epochs",fontsize=22)
    plt.ylabel("Loss",fontsize=22)
    plt.show()



if __name__ == "__main__":
    main()
