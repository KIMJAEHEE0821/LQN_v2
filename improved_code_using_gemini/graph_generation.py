# graph_generation.py
"""
Functions related to the generation of initial EPM bipartite graphs.
Includes combination generation helpers and the main graph generator.
"""

import igraph as ig
import itertools
from typing import List, Tuple, Iterator, Any, Dict # For type hinting

# --- Combination Generation Functions ---

def list_all_combinations_with_duplication(num_system: int, num_ancilla: int) -> Tuple[List[Tuple[Tuple[int, int], ...]], int]:
    """
    Generates all combinations of ordered pairs (i, j) where i != j,
    taken num_system at a time with repetition, representing system node connections.

    Parameters:
    -----------
    num_system : int
        Number of system nodes. Determines the length of the combinations.
    num_ancilla : int
        Number of ancilla nodes. Used to determine the total pool of target nodes.

    Returns:
    --------
    Tuple[List[Tuple[Tuple[int, int], ...]], int]
        A tuple containing:
        - A list of all combinations. Each combination is a tuple of length num_system,
          where each element is a pair (i, j) representing target node indices.
        - The total number of combinations generated.
    """
    p = num_system + num_ancilla
    if p < 2: # Need at least 2 nodes for pairs (i, j) where i != j
        return [], 0
    if num_system <= 0: # No system nodes, no combinations needed
        return [], 0

    vertices: List[int] = list(range(p))
    # Generate all ordered pairs (i, j) where i != j
    all_pairs: List[Tuple[int, int]] = list(itertools.permutations(vertices, 2))

    # Generate all combinations of these pairs, taking num_system pairs at a time,
    # with repetition allowed. This represents assigning two distinct target nodes
    # to each system node independently.
    all_combinations: List[Tuple[Tuple[int, int], ...]] = list(itertools.product(all_pairs, repeat=num_system))

    return all_combinations, len(all_combinations)

def generate_combinations(n: int) -> List[Tuple[int, ...]]:
    """
    Generates all possible non-empty combinations of elements from 0 to n-1.
    Used for determining ancilla node connections.

    Parameters:
    -----------
    n : int
        The number of elements (target nodes) to choose combinations from (typically num_system + num_ancilla).

    Returns:
    --------
    List[Tuple[int, ...]]
        A list of tuples, where each tuple represents a combination of target node indices.
        Includes combinations of length 1 up to n.
    """
    if n <= 0:
        return []

    all_combinations: List[Tuple[int, ...]] = []
    elements: List[int] = list(range(n))
    # Generate combinations for all possible sizes (from 1 to n)
    for i in range(1, n + 1):
        # combinations returns tuples
        combinations_of_size_i: List[Tuple[int, ...]] = list(itertools.combinations(elements, i))
        all_combinations.extend(combinations_of_size_i)

    return all_combinations

# --- Bipartite Graph Generation ---

def _create_bipartite_graph_structure(num_system: int, num_ancilla: int) -> Tuple[ig.Graph, List[str], List[str], List[str]]:
    """
    Creates the basic node structure and attributes for the EPM bipartite graph.

    Parameters:
    -----------
    num_system : int
    num_ancilla : int

    Returns:
    --------
    Tuple[ig.Graph, List[str], List[str], List[str]]
        A tuple containing:
        - The initialized igraph.Graph object.
        - List of system node names.
        - List of ancilla node names.
        - List of sculpting node names.
    """
    num_total_nodes_left = num_system + num_ancilla # System + Ancilla nodes
    num_total_nodes_right = num_total_nodes_left   # Sculpting nodes (typically same count)

    system_node_names = [f'S_{i}' for i in range(num_system)]
    ancilla_node_names = [f'A_{i}' for i in range(num_ancilla)]
    sculpting_node_names = [str(i) for i in range(num_total_nodes_right)] # Names are '0', '1', ...

    all_node_names = system_node_names + ancilla_node_names + sculpting_node_names
    num_all_nodes = len(all_node_names)

    G = ig.Graph()
    G.add_vertices(num_all_nodes)
    G.vs["name"] = all_node_names

    # Assign categories
    categories = ["system_nodes"] * num_system + ["ancilla_nodes"] * num_ancilla + ["sculpting_nodes"] * num_total_nodes_right
    G.vs["category"] = categories

    # Assign bipartite types (0 for left partition, 1 for right partition)
    bipartite_types = [0] * num_total_nodes_left + [1] * num_total_nodes_right
    G.vs["bipartite"] = bipartite_types

    return G, system_node_names, ancilla_node_names, sculpting_node_names

def _add_system_edges(G: ig.Graph, rb_comb: Tuple[Tuple[int, int], ...], num_system: int, num_ancilla: int) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Adds edges connecting system nodes to sculpting nodes based on the red-blue combination.

    Parameters:
    -----------
    G : igraph.Graph
        The graph object to add edges to.
    rb_comb : Tuple[Tuple[int, int], ...]
        A combination specifying pairs of target sculpting node indices for each system node.
        Length of rb_comb should be num_system.
    num_system : int
    num_ancilla : int

    Returns:
    --------
    Tuple[List[Tuple[int, int]], List[float]]
        A tuple containing the list of edges (source_idx, target_idx) and the list of corresponding weights.
    """
    edges = []
    edge_weights = []
    # Offset to get the index of the first sculpting node
    sculpting_node_offset = num_system + num_ancilla

    if len(rb_comb) != num_system:
         print(f"Warning: Length of rb_comb ({len(rb_comb)}) does not match num_system ({num_system}). Skipping system edge addition.")
         return [], []

    for sys_node_idx, target_pair in enumerate(rb_comb):
        # target_pair is (target_idx1, target_idx2) relative to sculpting nodes
        try:
            # Convert relative sculpting indices to absolute indices in the graph G
            red_sculpting_idx = sculpting_node_offset + target_pair[0]
            blue_sculpting_idx = sculpting_node_offset + target_pair[1]
        except IndexError:
             print(f"Warning: Invalid target pair format in rb_comb: {target_pair}. Skipping edge for system node {sys_node_idx}.")
             continue

        # Ensure indices are within the bounds of the graph vertices
        max_vertex_index = G.vcount() - 1
        if 0 <= sys_node_idx < num_system and \
           sculpting_node_offset <= red_sculpting_idx <= max_vertex_index and \
           sculpting_node_offset <= blue_sculpting_idx <= max_vertex_index:

            # Add edge for 'red' connection (weight 1.0)
            edges.append((sys_node_idx, red_sculpting_idx))
            edge_weights.append(1.0)
            # Add edge for 'blue' connection (weight 2.0)
            edges.append((sys_node_idx, blue_sculpting_idx))
            edge_weights.append(2.0)
        else:
            print(f"Warning: Node index out of bounds when adding system edge. Sys: {sys_node_idx}, Red: {red_sculpting_idx}, Blue: {blue_sculpting_idx}. Max index: {max_vertex_index}. Skipping.")

    return edges, edge_weights

def _add_ancilla_edges(G: ig.Graph, bl_comb: Tuple[Tuple[int, ...], ...], num_system: int, num_ancilla: int) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Adds edges connecting ancilla nodes to sculpting nodes based on the ancilla combination.

    Parameters:
    -----------
    G : igraph.Graph
        The graph object to add edges to.
    bl_comb : Tuple[Tuple[int, ...], ...]
        Specifies target sculpting node indices for each ancilla node.
        Length of bl_comb should be num_ancilla. Each element is a tuple of target indices.
    num_system : int
    num_ancilla : int

    Returns:
    --------
    Tuple[List[Tuple[int, int]], List[float]]
        A tuple containing the list of edges and their corresponding weights.
    """
    edges = []
    edge_weights = []
    # Index of the first ancilla node
    ancilla_node_offset = num_system
    # Index of the first sculpting node
    sculpting_node_offset = num_system + num_ancilla
    max_vertex_index = G.vcount() - 1

    if len(bl_comb) != num_ancilla:
         print(f"Warning: Length of bl_comb ({len(bl_comb)}) does not match num_ancilla ({num_ancilla}). Skipping ancilla edge addition.")
         return [], []

    for anc_idx_relative, target_indices_tuple in enumerate(bl_comb):
        # Absolute index of the current ancilla node
        ancilla_node_idx = ancilla_node_offset + anc_idx_relative

        for target_idx_relative in target_indices_tuple:
            # Convert relative sculpting index to absolute index
            try:
                sculpting_node_idx = sculpting_node_offset + target_idx_relative
            except IndexError: # Should not happen if target_indices_tuple contains valid indices
                 print(f"Warning: Invalid relative sculpting index {target_idx_relative} in bl_comb. Skipping edge.")
                 continue

            # Ensure indices are valid
            if ancilla_node_offset <= ancilla_node_idx < sculpting_node_offset and \
               sculpting_node_offset <= sculpting_node_idx <= max_vertex_index:
                 # Add edge with weight 3.0 for ancilla connection
                 edges.append((ancilla_node_idx, sculpting_node_idx))
                 edge_weights.append(3.0)
            else:
                 print(f"Warning: Node index out of bounds when adding ancilla edge. Anc: {ancilla_node_idx}, Sculp: {sculpting_node_idx}. Max index: {max_vertex_index}. Skipping.")

    return edges, edge_weights

def EPM_bipartite_graph_generator_igraph(num_system: int, num_ancilla: int) -> Iterator[ig.Graph]:
    """
    Generates EPM bipartite graphs using igraph (Refactored Version).

    Iterates through combinations of system and ancilla node connections
    to sculpting nodes, yielding valid graph structures.

    Parameters:
    -----------
    num_system : int
        Number of system nodes (non-negative).
    num_ancilla : int
        Number of ancilla nodes (non-negative).

    Yields:
    -------
    igraph.Graph
        Generated EPM bipartite graphs where all nodes have a degree of at least 2.

    Raises:
    -------
    ValueError:
        If num_system or num_ancilla are negative.
    """
    if num_system < 0 or num_ancilla < 0:
        raise ValueError("Number of system and ancilla nodes must be non-negative.")

    num_total_left = num_system + num_ancilla
    if num_total_left == 0: # Handle edge case of no system/ancilla nodes
         print("Warning: num_system and num_ancilla are both 0. No graphs generated.")
         return # Stop iteration

    # Generate system connection combinations
    red_blue_combinations, _ = list_all_combinations_with_duplication(num_system, num_ancilla)

    # Generate ancilla connection combinations
    ancilla_connection_targets = [()] # Default: empty tuple if no ancillas
    if num_ancilla > 0:
        # Get all possible subsets of sculpting nodes an ancilla *could* connect to
        single_ancilla_options = generate_combinations(num_total_left)
        if not single_ancilla_options:
             print("Warning: No valid connection options found for ancillas. No graphs will be generated with ancillas.")
             ancilla_connection_targets = [] # Prevent iteration if options are empty
        else:
             # Generate combinations for *each* ancilla node independently
             ancilla_connection_targets = list(itertools.product(single_ancilla_options, repeat=num_ancilla))

    # Main generation loop
    graph_count = 0
    yielded_count = 0
    for rb_comb in red_blue_combinations:
        # If num_ancilla is 0, this inner loop runs once with bl_comb = ()
        # If ancilla_connection_targets is empty, this loop is skipped
        for bl_comb in ancilla_connection_targets:
            graph_count += 1
            # Create the basic graph structure (nodes, names, categories, bipartite types)
            try:
                 G, _, _, _ = _create_bipartite_graph_structure(num_system, num_ancilla)
            except Exception as e:
                 print(f"Error creating graph structure for combination {graph_count}: {e}. Skipping.")
                 continue

            all_edges = []
            all_weights = []

            # Add system edges (if any systems exist)
            if num_system > 0:
                sys_edges, sys_weights = _add_system_edges(G, rb_comb, num_system, num_ancilla)
                all_edges.extend(sys_edges)
                all_weights.extend(sys_weights)

            # Add ancilla edges (if any ancillas exist)
            if num_ancilla > 0:
                # bl_comb is a tuple where each element corresponds to an ancilla
                # e.g., if num_ancilla=2, bl_comb = ( (targets_for_A0), (targets_for_A1) )
                anc_edges, anc_weights = _add_ancilla_edges(G, bl_comb, num_system, num_ancilla)
                all_edges.extend(anc_edges)
                all_weights.extend(anc_weights)

            # Add collected edges and weights to the graph
            if all_edges:
                try:
                    G.add_edges(all_edges)
                    G.es["weight"] = all_weights
                except Exception as e:
                     print(f"Error adding edges/weights for combination {graph_count}: {e}. Skipping.")
                     continue # Skip to next combination

            # Check degree constraint: yield the graph only if all nodes have degree >= 2
            if G.vcount() > 0: # Ensure graph is not empty
                 min_degree = min(G.degree()) if G.vcount() > 0 else 0
                 if min_degree >= 2:
                     yielded_count += 1
                     yield G
                 # else: # Optional: log skipped graphs
                 #    print(f"Graph {graph_count} skipped due to min degree {min_degree} < 2")
            # else: # Optional: log empty graph case if needed
            #    print(f"Graph {graph_count} is empty. Skipping.")

    print(f"Graph generation complete. Total combinations checked: {graph_count}. Graphs yielded (degree >= 2): {yielded_count}.")

