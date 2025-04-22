#########################################
#########################################
### Run this script to play the game  ###
#########################################
#########################################

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

import torch
import networkx as nx

from agent import DQNAgent
from catmouseenv import CatMouseEnv
from config import INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

input_dim, hidden_dims, output_dim = INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM

# Consistent layout for plot
graph_pos = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 0.4),
    3: (-1, 0.4),
    4: (0.6, -0.9),
    5: (-0.6, -0.9),
}

def draw_graph(G, cat_pos, mouse_pos):
    plt.clf()
    plt.figure(1, figsize=(6, 4))
    plt.cla()

    nx.draw(
        G,
        graph_pos,
        with_labels=True,
        node_color="lightgreen",
        node_size=888,
        edge_color="black",
        font_size=14,
    )


    plt.text(*graph_pos[cat_pos], '🐱', fontsize=24, ha='center', va='center', zorder=3)
    plt.text(*graph_pos[mouse_pos], '🐭', fontsize=24, ha='center', va='center', zorder=3)


    plt.title("Current Game State")
    plt.axis("off")
    plt.draw()
    plt.pause(0.5)  # Give time to render updates

    from matplotlib.lines import Line2D

    # Custom legend with emoji symbols
    legend_elements = [
        Line2D([0], [0], marker='$🐱$', markersize=12,
               linestyle='None', label='Cat', color = 'k'),
        Line2D([0], [0], marker='$🐭$', markersize=12,
               linestyle='None', label='Mouse', color = 'k')
    ]

    plt.legend(handles=legend_elements, loc='upper left', frameon=False)


def play_game(model_path: str, mode: str = "random", max_moves: int = 10):
    env = CatMouseEnv(mode=mode)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 4), (0, 5), (1, 2), (1, 3), (2, 4), (3, 5), (4, 5)])

    agent = DQNAgent(input_dim, hidden_dims, output_dim, epsilon=0.0)
    agent.load(model_path)
    agent.eval_mode()

    print(f"Game Start — Cat: {env.cat.position}, Mouse: {env.mouse.position}")
    draw_graph(G, env.cat.position, env.mouse.position)

    move_count = 0
    done = False

    while not done and move_count < max_moves:
        move_count += 1
        print(f"\nTurn {move_count}")

        # === Mouse (agent) moves ===
        mouse_neighbors = env.available_mouse_actions()
        mouse_action = agent.act(env._get_state(), mouse_neighbors, eval=True)
        env.mouse.position = mouse_action

        draw_graph(G, env.cat.position, env.mouse.position)
        print(f"Mouse moved to {mouse_action}")

        # === Cat (you) move ===
        cat_neighbors = env.graph.adj[env.cat.position]
        print(f"Cat at {env.cat.position}. Available moves: {cat_neighbors}")
        move_input = input("Enter node to move cat to (or 'q' to quit): ").strip()

        if move_input.lower() == "q":
            print("Game ended early.")
            break

        try:
            cat_move = int(move_input)
            if cat_move not in cat_neighbors:
                print("Invalid move. Try again.")
                continue
        except ValueError:
            print("Invalid input. Enter a valid node number.")
            continue

        env.cat.position = cat_move
        draw_graph(G, env.cat.position, env.mouse.position)

        if env.cat.position == env.mouse.position:
            print("\n You caught the mouse! Cat wins.")
            done = True
        elif move_count >= max_moves:
            print("\n Mouse survived! You lose.")
            done = True


if __name__ == "__main__":

    play_game(
        model_path="./saved_models/mouse_model_20250421_194945.pt",
        mode="random",
        max_moves=10,
    )
