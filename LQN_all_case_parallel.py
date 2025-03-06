import argparse
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from LQN_utils_state_save_parallel import *
import itertools
from copy import deepcopy

def main(num_system, num_ancilla, type):
    bigraph_result, digraph_result = graph_generation(num_system, num_ancilla, type)

    # Generate file names based on the input parameters
    bigraph_filename = f'bigraph_result_sys{num_system}_anc{num_ancilla}_type{type}.pkl'
    digraph_filename = f'digraph_result_sys{num_system}_anc{num_ancilla}_type{type}.pkl'

    # Save results to files
    with open(bigraph_filename, 'wb') as f:
        pickle.dump(bigraph_result, f)

    with open(digraph_filename, 'wb') as f:
        pickle.dump(digraph_result, f)

    print(f"Results saved to '{bigraph_filename}' and '{digraph_filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs based on given parameters.")
    parser.add_argument('--num_system', type=int, required=True, help='Number of systems')
    parser.add_argument('--num_ancilla', type=int, required=True, help='Number of ancillas')
    parser.add_argument('--type', type=int, required=True, help='Type of graph')

    args = parser.parse_args()

    main(args.num_system, args.num_ancilla, args.type)