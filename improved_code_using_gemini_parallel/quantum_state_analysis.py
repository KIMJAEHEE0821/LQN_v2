# quantum_state_analysis.py
"""
Functions for Perfect Matching analysis and initial Quantum State processing.
(Corresponds to Sections 1 and 2 of the description)
"""

import igraph as ig
import itertools
from itertools import combinations, permutations, groupby, chain
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import pickle
from copy import deepcopy
import hashlib
from collections import Counter, defaultdict
from functools import reduce
from math import gcd
from typing import List, Tuple, Dict, Any, Set, Optional, Union, Counter as CounterType, Iterable, Sequence, TypeAlias
import sympy as sp
from sympy.physics.quantum import Ket
from sympy import Add
from quantum_utils import *
import sys
from math import isclose
# ==============================================================================
# Section 1: Perfect Matching Analysis Functions
# ==============================================================================

def get_bipartite_sets(G: ig.Graph) -> Tuple[List[int], List[int]]:
    """
    Extracts the two node sets (partitions) of a bipartite graph.

    It first checks for an explicit 'bipartite' attribute (0 or 1).
    If not found, it falls back to using the 'category' attribute, grouping
    'system_nodes' and 'ancilla_nodes' as one set (U) and 'sculpting_nodes'
    as the other set (V).
    If neither attribute is present, it raises an error.

    Parameters:
    -----------
    G : igraph.Graph
        The input bipartite graph.

    Returns:
    --------
    Tuple[List[int], List[int]]
        A tuple containing two lists of node indices: (U, V).

    Raises:
    ------
    ValueError:
        If the graph doesn't have 'bipartite' or 'category' attributes
        to determine the partitions.
    """
    U: List[int] = []
    V: List[int] = []

    if 'bipartite' in G.vs.attributes():
        try:
            U = [v.index for v in G.vs if v['bipartite'] == 0]
            V = [v.index for v in G.vs if v['bipartite'] == 1]
        except KeyError:
            raise ValueError("Graph has 'bipartite' attribute but some nodes lack the value.")
    elif 'category' in G.vs.attributes():
        try:
            U = [v.index for v in G.vs if v['category'] in ['system_nodes', 'ancilla_nodes']]
            V = [v.index for v in G.vs if v['category'] == 'sculpting_nodes']
        except KeyError:
            raise ValueError("Graph has 'category' attribute but some nodes lack the value.")
    else:
        raise ValueError("Cannot determine bipartite sets. Graph lacks 'bipartite' or 'category' node attributes.")

    # Basic validation
    if not U or not V:
         print(f"Warning: One or both bipartite sets are empty (U: {len(U)}, V: {len(V)}).")
         # Depending on requirements, could raise ValueError here if sets must be non-empty
    if len(U) + len(V) != G.vcount():
         raise ValueError("Sum of nodes in bipartite sets does not match total vertex count.")
    # In many bipartite matching contexts, |U| == |V|, but not strictly required here yet
    # if len(U) != len(V):
    #     print(f"Warning: Bipartite sets have different sizes (|U|={len(U)}, |V|={len(V)}).")

    return U, V


def get_edge_weight(G: ig.Graph, u: int, v: int) -> Optional[float]:
    """
    Gets the weight of the edge between nodes u and v.

    Parameters:
    -----------
    G : igraph.Graph
        The graph.
    u : int
        Source node index.
    v : int
        Target node index.

    Returns:
    --------
    Optional[float]
        The weight of the edge if it exists and has a 'weight' attribute.
        Returns None if the edge doesn't exist or lacks a 'weight'.
        Returns 1.0 if the edge exists but 'weight' attribute is missing globally.
    """
    try:
        # Get edge ID, return None if edge doesn't exist
        eid = G.get_eid(u, v, directed=False, error=False)
        if eid == -1:
            return None

        # Check if 'weight' attribute exists for edges
        if 'weight' in G.es.attributes():
            weight = G.es[eid]['weight']
            # Ensure weight is float or convertible to float
            try:
                 return float(weight)
            except (ValueError, TypeError):
                 print(f"Warning: Edge ({u},{v}) weight '{weight}' is not a number. Returning None.")
                 return None
        else:
            # If no 'weight' attribute exists on any edge, assume default weight 1.0
            # print("Warning: Edge 'weight' attribute not found. Assuming default weight 1.0.")
            return 1.0

    except Exception as e:
        print(f"Error getting edge weight for ({u},{v}): {e}")
        return None


def is_perfect_matching(G: ig.Graph, U: List[int], V: List[int], matching_edges: List[Tuple[int, int, float]]) -> bool:
    """
    Checks if a given set of edges constitutes a perfect matching in the bipartite graph G.

    A perfect matching must cover all nodes in the smaller partition, and if the
    partitions have equal size, it must cover all nodes in the graph.

    Parameters:
    -----------
    G : igraph.Graph
        The bipartite graph.
    U : List[int]
        List of node indices in the first partition.
    V : List[int]
        List of node indices in the second partition.
    matching_edges : List[Tuple[int, int, float]]
        The list of edges in the potential matching. Each tuple is (u, v, weight).

    Returns:
    --------
    bool
        True if the edges form a perfect matching, False otherwise.
    """
    # A perfect matching exists only if |U| == |V| in most standard definitions.
    # It must contain |U| (or |V|) edges.
    if len(U) != len(V):
        # print("Warning: Cannot have a perfect matching in a bipartite graph with unequal partitions.")
        # Depending on the exact definition needed, might allow matching all nodes of the smaller set.
        # For now, assume standard definition requiring |U| == |V|.
        return False

    if len(matching_edges) != len(U):
        return False # Incorrect number of edges for a perfect matching

    # Check if all nodes in U and V are covered exactly once
    matched_nodes_U = set()
    matched_nodes_V = set()
    for u, v, _ in matching_edges:
        # Ensure the edge is actually between U and V
        if (u in U and v in V):
            matched_nodes_U.add(u)
            matched_nodes_V.add(v)
        elif (v in U and u in V):
            matched_nodes_U.add(v)
            matched_nodes_V.add(u)
        else:
            # Edge does not connect U and V
            return False

    # Check if all nodes are covered and no node is matched twice
    return len(matched_nodes_U) == len(U) and len(matched_nodes_V) == len(V)


def dfs_all_matchings(
    G: ig.Graph,
    U: List[int],
    V: List[int],
    u_index: int,
    current_matching: List[Tuple[int, int, float]],
    matched_V: Dict[int, bool],
    all_matchings: List[List[Tuple[int, int, float]]],
    all_weight_strings: List[str]
) -> None:
    """
    Recursive Depth-First Search function to find all perfect matchings.
    Edges with weight 3.0 are IGNORED when generating the weight string.

    Parameters:
    -----------
    G : igraph.Graph
        The bipartite graph.
    U : List[int]
        Nodes in the first partition (nodes to be matched).
    V : List[int]
        Nodes in the second partition (potential partners).
    u_index : int
        The index of the current node in U being considered for matching.
    current_matching : List[Tuple[int, int, float]]
        The list of edges forming the matching being built.
    matched_V : Dict[int, bool]
        A dictionary tracking which nodes in V are already matched in the current path.
    all_matchings : List[List[Tuple[int, int, float]]]
        List to store all found perfect matchings (as lists of edges).
    all_weight_strings : List[str]
        List to store the corresponding weight strings (state representations, potentially truncated)
        for each perfect matching.
    """
    # Base case: If all nodes in U have been matched, we found a perfect matching
    if u_index == len(U):
        all_matchings.append(current_matching[:]) # Store a copy

        # Generate the weight string (state representation)
        # Mapping: weight 1.0 -> '0', weight 2.0 -> '1', weight 3.0 -> IGNORED
        # Sort edges by the index of the node in U to ensure consistent string order
        # Note: Ensure U contains unique indices corresponding to one partition for sorting key to work reliably.
        # If U can contain nodes from both partitions (less likely for bipartite matching context), adjust sorting key.
        try:
             # Assuming U corresponds to the indices that define the order for the state string
             u_indices_map = {node_idx: i for i, node_idx in enumerate(U)}
             # Sort based on the order defined by U
             sorted_matching = sorted(current_matching, key=lambda edge: u_indices_map.get(edge[0], float('inf')))
        except Exception as e_sort:
             print(f"Warning: Error during sorting matching edges based on U: {e_sort}. Using unsorted matching.")
             sorted_matching = current_matching # Fallback to unsorted if error

        weights_list = []
        for u_matched, v_matched, weight in sorted_matching:
            if weight == 1.0:
                weights_list.append('0')
            elif weight == 2.0:
                weights_list.append('1')
            elif weight == 3.0:
                # --- 여기가 수정된 부분 ---
                # Ignore edges with weight 3.0 (assumed to be ancilla)
                pass # 가중치 3.0을 가진 엣지는 상태 문자열 생성 시 무시합니다.
                # --- 수정 끝 ---
            else:
                # Handle unexpected weights if necessary
                print(f"Warning: Unexpected weight {weight} in matching for edge ({u_matched},{v_matched}). Ignoring edge in state string.")
                # Or assign a default/error character, or raise an error

        matching_key = ''.join(weights_list)
        all_weight_strings.append(matching_key) # Ancilla 정보가 제외된 문자열이 저장됩니다.
        return

    # Current node from partition U to match
    # Ensure u_index is within bounds of U
    if u_index >= len(U):
         # This case should ideally not be reached if base case is correct
         print(f"Warning: u_index {u_index} out of bounds for U (len {len(U)}). Aborting path.")
         return
    u = U[u_index]

    # Iterate through potential partners in partition V
    for v in V:
        # Check if v is already matched in the current path
        if not matched_V.get(v, False): # Use .get for safety
            # Check if an edge exists between u and v and get its weight
            # Ensure get_edge_weight is defined and handles non-existent edges returning None
            weight = get_edge_weight(G, u, v)
            if weight is not None:
                # Add edge (u, v, weight) to the current matching
                current_matching.append((u, v, weight))
                matched_V[v] = True # Mark v as matched

                # Recursively call for the next node in U
                dfs_all_matchings(G, U, V, u_index + 1, current_matching, matched_V, all_matchings, all_weight_strings)

                # Backtrack: Remove the edge and unmark v
                matched_V[v] = False
                current_matching.pop()


def find_all_perfect_matchings(G: ig.Graph) -> Tuple[List[List[Tuple[int, int, float]]], List[str]]:
    """
    Finds all perfect matchings in a bipartite graph using DFS.

    Parameters:
    -----------
    G : igraph.Graph
        The input bipartite graph.

    Returns:
    --------
    Tuple[List[List[Tuple[int, int, float]]], List[str]]
        A tuple containing:
        - A list where each element is a perfect matching (represented as a list of edges).
        - A list of corresponding weight strings ('0'/'1'/'2') for each perfect matching.

    Raises:
    ------
    ValueError:
        If the graph is not bipartite or has unequal partitions (required for perfect matching).
    """
    try:
        U, V = get_bipartite_sets(G)
    except ValueError as e:
        print(f"Cannot find perfect matchings: {e}")
        return [], []

    # Standard perfect matching requires partitions of equal size
    if len(U) != len(V):
        # print(f"Graph partitions are unequal (|U|={len(U)}, |V|={len(V)}). Cannot find perfect matching.")
        return [], []
    if not U: # If partitions are empty
        return [], []


    all_matchings: List[List[Tuple[int, int, float]]] = []
    all_weight_strings: List[str] = []
    matched_V: Dict[int, bool] = {v_node: False for v_node in V} # Track matched status of V nodes

    # Start DFS from the first node in U
    dfs_all_matchings(G, U, V, 0, [], matched_V, all_matchings, all_weight_strings)

    return all_matchings, all_weight_strings


def find_unused_edges(G: ig.Graph, perfect_matchings: List[List[Tuple[int, int, float]]]) -> List[Tuple[int, int]]:
    """
    Finds edges in the graph G that are not part of any of the provided perfect matchings.

    Parameters:
    -----------
    G : igraph.Graph
        The bipartite graph.
    perfect_matchings : List[List[Tuple[int, int, float]]]
        A list of perfect matchings found for graph G.

    Returns:
    --------
    List[Tuple[int, int]]
        A list of edges (source_idx, target_idx) present in G but not used
        in any of the provided perfect matchings.
    """
    if not perfect_matchings: # If no perfect matchings were found, all edges are technically "unused" in matchings
         # Depending on the goal, might return all edges or an empty list.
         # Let's return all edges in this case.
         # return [(e.source, e.target) for e in G.es]
         # Or, if the context implies only edges *could* be part of a PM, return empty.
         # Let's assume the check is meaningful only if PMs exist.
         return [] # Or handle as needed if no PMs exist

    # Create a set of all edges used in *any* perfect matching
    # Store edges as frozensets to handle directionality {(u, v)} == {(v, u)}
    used_edges_set: Set[frozenset] = set()
    for matching in perfect_matchings:
        for u, v, _ in matching:
            used_edges_set.add(frozenset([u, v]))

    # Find edges in G that are not in the used_edges_set
    unused_edges: List[Tuple[int, int]] = []
    for edge in G.es:
        edge_nodes = frozenset([edge.source, edge.target])
        if edge_nodes not in used_edges_set:
            unused_edges.append((edge.source, edge.target))

    return unused_edges


def find_pm_of_bigraph(graph_list: List[ig.Graph]) -> List[List[Any]]:
    """
    Processes a list of bipartite graphs to find perfect matchings and associated states.

    Filters results based on conditions:
    1. At least two perfect matchings must exist.
    2. No edges should be left unused by the set of all perfect matchings.

    Normalizes the coefficients (counts) of the resulting states using GCD.

    Parameters:
    -----------
    graph_list : List[igraph.Graph]
        A list of igraph.Graph objects (assumed to be bipartite).

    Returns:
    --------
    List[List[Any]]
        A list where each inner list corresponds to a valid graph and contains:
        [
            CounterType[str],  # Counter of weight strings (quantum states) with normalized coefficients
            List[List[Tuple[int, int, float]]], # List of all perfect matchings found
            ig.Graph,          # The original graph object
            int                # Index of the graph in the original input list
        ]
        Only includes entries for graphs satisfying the filtering conditions.
    """
    save_fw_results: List[List[Any]] = []
    processed_count = 0
    valid_count = 0
    error_count = 0

    for i, G in enumerate(graph_list):
        processed_count += 1
        try:
            # Find all perfect matchings and their weight strings
            perfect_matching_list, weight_strings = find_all_perfect_matchings(G)

            # Check filtering conditions
            # Condition 1: Must have at least 2 perfect matchings
            condition1_met = len(perfect_matching_list) >= 2

            # Condition 2: No unused edges among all perfect matchings found
            # Only check if condition 1 is met and PMs were actually found
            condition2_met = False
            if condition1_met:
                unused = find_unused_edges(G, perfect_matching_list)
                condition2_met = not unused # True if the list of unused edges is empty

            # If both conditions are met, process the results
            if condition1_met and condition2_met:
                valid_count += 1
                # Create a counter for the weight strings
                state_counter: CounterType[str] = Counter(weight_strings)

                # Normalize coefficients using GCD
                counts = list(state_counter.values())
                if len(counts) > 0:
                    # Calculate GCD of all counts
                    try:
                         # reduce might fail on an empty list, len(counts)>0 handles this
                         # gcd(0, x) = x, gcd(0, 0) = 0. Handle case of single count.
                         if len(counts) == 1:
                              # No normalization needed for a single state type
                              pass
                         else:
                              # Calculate GCD only if there are multiple counts > 0
                              non_zero_counts = [c for c in counts if c > 0]
                              if len(non_zero_counts) >= 2:
                                   common_divisor = reduce(gcd, non_zero_counts)
                                   if common_divisor > 1:
                                        # Divide each count by the GCD
                                        for state in state_counter:
                                             state_counter[state] //= common_divisor
                              # If only one non-zero count, or all non-zero counts are 1, GCD is 1, no change needed.

                    except Exception as e_gcd:
                        print(f"Warning: Error calculating or applying GCD for graph index {i}: {e_gcd}")


                save_fw_results.append([
                    state_counter,
                    perfect_matching_list,
                    G,
                    i # Original index
                ])

        except Exception as e:
            error_count += 1
            print(f"Error processing graph index {i} for perfect matchings: {e}")
            # Optionally add more details, e.g., G.summary()

    print(f"Perfect matching analysis complete. Processed: {processed_count}, Valid (met conditions): {valid_count}, Errors: {error_count}")
    return save_fw_results


# ==============================================================================
# Section 2: Quantum State Generation & Initial Processing Functions
# ==============================================================================

def remove_same_state(save_fw_results: List[List[Any]]) -> List[List[Any]]:
    """
    Removes duplicate entries from the results based on the state Counter.

    Keeps only the first occurrence of each unique state Counter.
    The Counter includes both the states (keys) and their normalized coefficients (values).

    Parameters:
    -----------
    save_fw_results : List[List[Any]]
        The list generated by `find_pm_of_bigraph`. Each inner list is
        [Counter, matchings, graph, index].

    Returns:
    --------
    List[List[Any]]
        A filtered list containing only entries with unique state Counters.
    """
    seen_counters: Set[Tuple[Tuple[str, int], ...]] = set()
    unique_results: List[List[Any]] = []
    input_count = len(save_fw_results)

    for result_entry in save_fw_results:
        state_counter = result_entry[0]
        if not isinstance(state_counter, Counter):
             print(f"Warning: Expected a Counter object at index 0, got {type(state_counter)}. Skipping entry.")
             continue

        # Create a canonical representation of the counter for checking uniqueness.
        # Sort items (state, count) by state string to ensure consistent order.
        counter_tuple: Tuple[Tuple[str, int], ...] = tuple(sorted(state_counter.items()))

        # If this specific counter content hasn't been seen before, add it
        if counter_tuple not in seen_counters:
            seen_counters.add(counter_tuple)
            unique_results.append(result_entry)

    output_count = len(unique_results)
    print(f"Removed {input_count - output_count} duplicate states based on Counter content. Kept {output_count} unique entries.")
    return unique_results


def process_graph_dict(graph_dict: Dict[str, List[ig.Graph]]) -> Dict[str, List[List[Any]]]:
    """
    Processes a dictionary of graph groups. For each group (list of graphs),
    it finds perfect matchings, generates states, filters based on conditions,
    normalizes coefficients, and removes duplicates based on the state Counter.

    Parameters:
    -----------
    graph_dict : Dict[str, List[igraph.Graph]]
        Dictionary mapping hash keys (from structural isomorphism) to lists
        of potentially weighted-isomorphic igraph.Graph objects.

    Returns:
    --------
    Dict[str, List[List[Any]]]
        Dictionary mapping the same hash keys to the processed results.
        Each value is a list of unique state entries for that group, where each entry is
        [Counter, matchings, graph, original_index_within_group_list].
        Returns an empty list for a key if no graphs in that group satisfy the conditions
        or result in unique states.
    """
    result_dict: Dict[str, List[List[Any]]] = {}
    total_unique_states_found = 0

    print(f"Processing {len(graph_dict)} graph groups for quantum states...")
    for hash_key, graph_list in graph_dict.items():
        if not graph_list:
            result_dict[hash_key] = []
            continue

        # 1. Find perfect matchings and initial states, filtering by conditions (>=2 PMs, no unused edges)
        # The index 'i' returned inside fw_results is relative to the input graph_list for this key
        fw_results = find_pm_of_bigraph(graph_list)

        # 2. Remove entries that produced the exact same state counter
        unique_state_results = remove_same_state(fw_results)

        # Store the results for this hash key
        result_dict[hash_key] = unique_state_results
        total_unique_states_found += len(unique_state_results)
        # print(f"Group '{hash_key}': Found {len(unique_state_results)} unique states after initial processing.")


    print(f"Quantum state initial processing complete. Total unique states found (before further filtering): {total_unique_states_found}")
    return result_dict

# quantum_state_analysis.py (Continued)
# (Append this code to the content from quantum_state_analysis_part1_code_ko)


# ==============================================================================
# Section 3: 
# ==============================================================================

def check_quantum_states_exist(result_dict, target_states, hash_key=None):
    """
    Check if all specified quantum states exist in the results.
    
    Parameters:
    -----------
    result_dict : dict
        Result dictionary returned from process_graph_dict() function
    target_states : list or str
        List of quantum states to search for (e.g. ['0101', '1010']) or a single state as string
    hash_key : str, optional
        Specific hash key to search within (default: None, search all hashes)
        
    Returns:
    --------
    list
        [(hash_key, {state1: coefficient1, state2: coefficient2, ...}, graph_index), ...] 
        List of results where all target states exist:
        - hash_key: Hash key where the states were found
        - state_coefficients: Dictionary of states and their coefficients
        - graph_index: Index in the original graph list
    """
    results = []
    
    # Convert single state string to list for consistent processing
    if isinstance(target_states, str):
        target_states = [target_states]
    
    # Determine which hash keys to search
    if hash_key is not None:
        if hash_key not in result_dict:
            return []
        hash_keys = [hash_key]
    else:
        hash_keys = result_dict.keys()
    
    # Search through each hash key
    for key in hash_keys:
        for state_data in result_dict[key]:
            counter = state_data[0]  # State coefficient Counter
            graph_index = state_data[3]  # Graph index
            
            # Check if all target states exist in this counter
            all_states_exist = all(state in counter for state in target_states)
            
            if all_states_exist:
                # Create dictionary of states and their coefficients
                state_coefficients = {state: counter[state] for state in target_states}
                results.append((key, state_coefficients, graph_index))
    
    return results


# --- Import the apply_bit_flip function from the other file ---
try:
    from quantum_utils import apply_bit_flip
except ImportError:
    print("ERROR: Failed to import 'apply_bit_flip' from 'quantum_utils.py'.", file=sys.stderr)
    # Optionally, terminate the program here or define a dummy function.
    # Example: def apply_bit_flip(*args, **kwargs): raise RuntimeError("apply_bit_flip not loaded")
    raise # Re-raise the error to stop the program

# --- igraph import handling block (located here as check_quantum_states_with_bit_flips uses it) ---
try:
    import igraph as ig
    IGRAPH_INSTALLED = True
except ImportError:
    class DummyGraph: pass # Dummy class to use if igraph is not installed
    ig = type("igraph", (object,), {"Graph": DummyGraph})() # Create a mock igraph.Graph module/class
    IGRAPH_INSTALLED = False
    print("Warning: 'igraph' library not found. Graph object type checks might be skipped.", file=sys.stderr)

# --- Define CounterType Alias (located here as check_quantum_states_with_bit_flips uses it) ---
CounterType: TypeAlias = Counter


 # This function assumes the necessary imports and definitions are present
# in the same file (state_analyzer.py context from previous examples):
# import sys, Counter, chain, combinations, List, Dict, Any, Optional, Union, Tuple, TypeAlias
# from quantum_utils import apply_bit_flip (or fallback)
# igraph import handling (ig, IGRAPH_INSTALLED, DummyGraph)
# CounterType alias

# def check_quantum_states_with_bit_flips(
#     result_dict: Dict[str, List[List[Any]]],
#     target_states: Union[str, List[str]],
#     bit_flip_positions: Optional[List[int]] = None,
#     hash_key: Optional[str] = None,
#     exact_num_states: Optional[int] = None
# ) -> List[Tuple[str, Any, int, CounterType[str], List[int], Dict[str, int]]]: # Return type uses Any for graph if igraph optional
#     """
#     Checks if quantum states exist in Counters within result_dict,
#     considering possible bit flips at specified positions. Finds only the *first*
#     successful bit flip combination for each Counter entry.
#     Optionally filters based on the exact number of states in the Counter.
#     Includes enhanced checks for data structure and logs unexpected errors.

#     Parameters:
#     -----------
#     result_dict : dict
#         Result dictionary. Assumed structure:
#         {'hash': [[Counter_obj, data1, graph_obj, graph_idx], ...], ...}
#         **NOTE:** Assumes Counter at index 0, graph object at index 2,
#                  graph index (int) at index 3. Adjust indices constants if needed.
#     target_states : Union[str, List[str]]
#         Quantum state(s) to search for (e.g., '010' or ['010', '111']).
#     bit_flip_positions : Optional[List[int]], default=None
#         List of 0-based positions where bit flips are allowed.
#         If None, all combinations of flips across the longest target state's
#         length are tried (can be computationally expensive).
#     hash_key : Optional[str], default=None
#         Specific hash key to search within result_dict. If None, searches all keys.
#     exact_num_states : Optional[int], default=None
#         If provided, only inspects Counters having exactly this number of states.

#     Returns:
#     --------
#     List[Tuple[str, Any, int, CounterType[str], List[int], Dict[str, int]]]
#         List of tuples for successful matches (max one per input Counter entry):
#         (hash_key, graph_object, graph_index, original_counter,
#          applied_flips_list, found_flipped_states_with_coeffs_dict)
#         The type of graph_object depends on whether 'igraph' is installed and used.

#     Raises:
#     -------
#     TypeError
#         If `target_states` is not a string or list of strings, or if elements
#         within the list are not strings.
#     """
#     results = []

#     # --- Input Processing: target_states ---
#     local_target_states: List[str] = []
#     if isinstance(target_states, str):
#         if target_states: local_target_states = [target_states]
#     elif isinstance(target_states, list):
#         if all(isinstance(s, str) for s in target_states):
#             local_target_states = [s for s in target_states if s] # Keep non-empty strings
#         else:
#             raise TypeError("If target_states is a list, all elements must be strings.")
#     else:
#         raise TypeError("target_states must be a string or a list of strings.")

#     if not local_target_states:
#          print("Warning: No valid non-empty target states provided to search for.", file=sys.stderr)
#          return []

#     # --- Input Processing: search_keys ---
#     search_keys: List[str] = []
#     if hash_key is not None:
#         if hash_key in result_dict:
#             search_keys = [hash_key]
#         else:
#             print(f"Warning: Specified hash_key '{hash_key}' not found in result_dict.", file=sys.stderr)
#             return []
#     else:
#         search_keys = list(result_dict.keys())
#     if not search_keys:
#         print("Warning: No keys to search in result_dict.", file=sys.stderr)
#         return []

#     # --- Input Processing: bit_flip_positions ---
#     actual_bit_flip_positions: List[int] = []
#     if bit_flip_positions is None:
#         try:
#             max_length = max(len(state) for state in local_target_states) if local_target_states else 0
#             if max_length > 0:
#                  actual_bit_flip_positions = list(range(max_length))
#         except ValueError:
#              print("Warning: Could not determine max length for default bit flips.", file=sys.stderr)
#              actual_bit_flip_positions = []
#     elif isinstance(bit_flip_positions, list):
#          valid_positions = [p for p in bit_flip_positions if isinstance(p, int) and p >= 0]
#          if len(valid_positions) != len(bit_flip_positions):
#              print("Warning: Some invalid values (non-integer or negative) removed from bit_flip_positions.", file=sys.stderr)
#          actual_bit_flip_positions = sorted(list(set(valid_positions))) # Unique, sorted, non-negative ints
#     else:
#          print("Warning: Invalid type for bit_flip_positions. No bit flips will be considered.", file=sys.stderr)
#          actual_bit_flip_positions = [] # Ensure it's a list

#     # --- Generate Bit Flip Combinations ---
#     all_combinations = list(chain.from_iterable(
#         combinations(actual_bit_flip_positions, r) for r in range(len(actual_bit_flip_positions) + 1)
#     ))
#     if not all_combinations:
#         all_combinations = [()] # Represents the "no flip" case

#     # --- Define Indices (Constants for clarity) ---
#     COUNTER_IDX = 0
#     GRAPH_OBJ_IDX = 2
#     GRAPH_IDX_IDX = 3
#     MIN_DATA_LEN = max(COUNTER_IDX, GRAPH_OBJ_IDX, GRAPH_IDX_IDX) + 1

#     # --- Main Processing Loop ---
#     for key in search_keys:
#         if key not in result_dict or not isinstance(result_dict[key], list):
#             print(f"Warning: Skipping key '{key}'. Missing or invalid data format (expected list).", file=sys.stderr)
#             continue

#         for i, state_data in enumerate(result_dict[key]):
#             # --- Structure and Type Validation for each item ---
#             if not isinstance(state_data, (list, tuple)) or len(state_data) < MIN_DATA_LEN:
#                 print(f"Warning: Skipping item {i} for key '{key}'. Invalid structure or length < {MIN_DATA_LEN}.", file=sys.stderr)
#                 continue

#             # Validate Counter
#             counter = state_data[COUNTER_IDX]
#             if not isinstance(counter, Counter):
#                 print(f"Warning: Skipping item {i} for key '{key}'. Expected Counter at index {COUNTER_IDX}, found {type(counter).__name__}.", file=sys.stderr)
#                 continue

#             # Validate Graph Object (conditionally if igraph installed)
#             graph_obj = state_data[GRAPH_OBJ_IDX]
#             if IGRAPH_INSTALLED and not isinstance(graph_obj, ig.Graph):
#                  print(f"Warning: Skipping item {i} for key '{key}'. Expected igraph.Graph at index {GRAPH_OBJ_IDX}, found {type(graph_obj).__name__}.", file=sys.stderr)
#                  continue

#             # Validate Graph Index
#             graph_idx = state_data[GRAPH_IDX_IDX]
#             if not isinstance(graph_idx, int):
#                  print(f"Warning: Skipping item {i} for key '{key}'. Expected int at index {GRAPH_IDX_IDX}, found {type(graph_idx).__name__}.", file=sys.stderr)
#                  continue

#             # --- Exact State Count Filter ---
#             if exact_num_states is not None and len(counter) != exact_num_states:
#                 continue # Skip if the number of states doesn't match exactly

#             # --- Iterate through Bit Flip Combinations ---
#             # This inner loop will now stop for the current state_data
#             # as soon as the first valid flip combination is found.
#             for bit_positions_tuple in all_combinations:
#                 bit_positions = list(bit_positions_tuple) # Use list version

#                 try:
#                     # Apply flips to all target states for this combination
#                     flipped_targets = [apply_bit_flip(state, bit_positions) for state in local_target_states]

#                     # Check if ALL flipped target states exist in the current counter
#                     all_states_exist = all(flipped_state in counter for flipped_state in flipped_targets)

#                     if all_states_exist:
#                         # Retrieve coefficients for the found states
#                         state_coefficients = {state: counter[state] for state in flipped_targets}

#                         # Construct and append the result tuple
#                         result_tuple = (
#                             key,
#                             graph_obj,
#                             graph_idx,
#                             counter,
#                             bit_positions, # List of positions flipped for this match
#                             state_coefficients
#                         )
#                         results.append(result_tuple)

#                         # --- BREAK ADDED ---
#                         # Stop searching for other flip combinations for THIS state_data item
#                         # once the first successful one is found.
#                         break
#                         # --- END OF ADDED BREAK ---

#                 except Exception as e:
#                     # Catch unexpected errors during flip application or check
#                     print(f"Warning: Unexpected error during processing for key '{key}', item {i}, flips {bit_positions}: {e}. Skipping this combination.", file=sys.stderr)
#                     # Continue to the next flip combination even if one fails
#                     continue

#             # End of loop for bit_positions_tuple for the current state_data
#         # End of loop for state_data in result_dict[key]
#     # End of loop for key in search_keys

#     return results


# --- Conditional igraph import ---
try:
    import igraph as ig
    IGRAPH_INSTALLED = True
except ImportError:
    IGRAPH_INSTALLED = False
    # Define a placeholder if igraph is not essential for the core logic
    # Or ensure the graph object handling doesn't strictly require ig.Graph type
    class ig: # Basic placeholder if needed
        class Graph: pass
    print("Warning: igraph library not found. Graph object type validation might be affected.", file=sys.stderr)



def check_quantum_states_with_bit_flips(
    result_dict: Dict[str, List[List[Any]]],
    # ===> Unified parameter <===
    targets: Union[str, List[str], Dict[str, int]],
    # === Remaining parameters ===
    bit_flip_positions: Optional[List[int]] = None,
    hash_key: Optional[str] = None,
    exact_num_states: Optional[int] = None
) -> List[Tuple[str, Any, int, Any, CounterType[str], List[int], Dict[str, int]]]:
    """
    Checks if quantum states exist in Counters within result_dict,
    considering possible bit flips. If targets is a dictionary, it also checks
    for matching coefficients. Finds only the *first* successful bit flip
    combination for each Counter entry. Optionally filters based on the
    exact number of states in the Counter.

    Parameters:
    -----------
    result_dict : dict
        Result dictionary. Assumed structure:
        {'hash': [[Counter, pm_data, graph_obj, graph_idx], ...], ...}
    targets : Union[str, List[str], Dict[str, int]]
        - If str: The single target quantum state string to search for.
        - If List[str]: A list of target state strings. Checks if *all* exist.
        - If Dict[str, int]: A dictionary mapping state strings to required integer
          coefficients. Checks if all states exist *and* have the specified coefficients.
    bit_flip_positions : Optional[List[int]], default=None
        List of 0-based positions where bit flips are allowed in target states.
        If None, all positions up to the longest target state length are considered.
    hash_key : Optional[str], default=None
        Specific hash key to search within result_dict. If None, searches all keys.
    exact_num_states : Optional[int], default=None
        If provided, only inspects Counters having exactly this number of states.

    Returns:
    --------
    List[Tuple[str, Any, int, Any, CounterType[str], List[int], Dict[str, int]]]
        List of tuples for successful matches (max one per input Counter entry):
        (hash_key, graph_object, graph_index, perfect_matching_data,
         original_counter, applied_flips_list, # List of positions flipped
         found_flipped_states_with_coeffs_dict) # Dict showing found states and their actual coeffs

    Raises:
    -------
    TypeError
        If `targets` has an invalid type or format.
    """
    results = []
    local_target_states: List[str] = []
    valid_target_coeffs: Optional[Dict[str, int]] = None # Stores coefficient check requirements

    # --- Input Type Processing ---
    if isinstance(targets, str):
        if targets: # Non-empty string
            local_target_states = [targets]
        # valid_target_coeffs remains None
    elif isinstance(targets, list):
        if all(isinstance(s, str) for s in targets):
            local_target_states = [s for s in targets if s] # Only non-empty strings
        else:
            raise TypeError("If targets is a list, all elements must be non-empty strings.")
        # valid_target_coeffs remains None
    elif isinstance(targets, dict):
        valid_target_coeffs = {}
        temp_target_states = []
        all_valid = True
        for state, coeff in targets.items():
            if not isinstance(state, str) or not state:
                print(f"Warning: Invalid state key '{state}' in targets dictionary. Skipping.", file=sys.stderr)
                all_valid = False; break
            if not isinstance(coeff, int):
                print(f"Warning: Coefficient for state '{state}' ({coeff}) must be an integer. Skipping.", file=sys.stderr)
                all_valid = False; break
            temp_target_states.append(state)
            valid_target_coeffs[state] = coeff # Add only if valid

        if all_valid and temp_target_states:
            local_target_states = temp_target_states # Set state list if valid
        else:
            print("Warning: Invalid format in targets dictionary. No states will be searched.", file=sys.stderr)
            return [] # Return empty list if invalid
    else:
        raise TypeError("targets must be a string, a list of strings, or a dictionary[str, int].")

    if not local_target_states:
         print("Warning: No valid non-empty target states derived to search for.", file=sys.stderr)
         return []

    # --- Input Processing: search_keys ---
    search_keys: List[str] = []
    if hash_key is not None:
        if hash_key in result_dict: search_keys = [hash_key]
        else: print(f"Warning: Specified hash_key '{hash_key}' not found.", file=sys.stderr); return []
    else: search_keys = list(result_dict.keys())
    if not search_keys: print("Warning: No keys to search in result_dict.", file=sys.stderr); return []

    # --- Input Processing: bit_flip_positions ---
    actual_bit_flip_positions: List[int] = []
    max_len_target = 0
    if local_target_states: # Ensure list is not empty before finding max length
         try:
             max_len_target = max(len(s) for s in local_target_states)
         except ValueError: # Handles empty strings if they somehow passed validation
             max_len_target = 0

    if bit_flip_positions is None:
        if max_len_target > 0:
            actual_bit_flip_positions = list(range(max_len_target))
    elif isinstance(bit_flip_positions, list):
         valid_positions = [p for p in bit_flip_positions if isinstance(p, int) and p >= 0]
         if len(valid_positions) != len(bit_flip_positions): print("Warning: Invalid values removed from bit_flip_positions.", file=sys.stderr)
         actual_bit_flip_positions = sorted(list(set(valid_positions)))
         # Optional check: ensure positions are within bounds of target state lengths
         if actual_bit_flip_positions and max_len_target > 0 and actual_bit_flip_positions[-1] >= max_len_target:
              print(f"Warning: Some bit_flip_positions might be out of bounds for the shortest target state.", file=sys.stderr)
    else:
         print("Warning: Invalid type for bit_flip_positions. No bit flips will be considered.", file=sys.stderr)

    # --- Generate Bit Flip Combinations ---
    all_combinations = list(chain.from_iterable(
        combinations(actual_bit_flip_positions, r) for r in range(len(actual_bit_flip_positions) + 1)
    ))
    if not all_combinations: all_combinations = [()] # Include the no-flip case

    # --- Define Indices ---
    COUNTER_IDX, PERFECT_MATCHING_IDX, GRAPH_OBJ_IDX, GRAPH_IDX_IDX = 0, 1, 2, 3
    MIN_DATA_LEN = 4 # Ensure required indices exist

    # --- Main Processing Loop ---
    for key in search_keys:
        if key not in result_dict or not isinstance(result_dict[key], list): continue

        for i, state_data in enumerate(result_dict[key]):
            # --- Structure and Type Validation ---
            if not isinstance(state_data, (list, tuple)) or len(state_data) < MIN_DATA_LEN: continue
            counter = state_data[COUNTER_IDX]
            if not isinstance(counter, Counter): continue
            perfect_matching_data = state_data[PERFECT_MATCHING_IDX]
            graph_obj = state_data[GRAPH_OBJ_IDX]
            graph_idx = state_data[GRAPH_IDX_IDX]
            if not isinstance(graph_idx, int): continue
            # (igraph validation if needed)

            # --- Exact State Count Filter ---
            if exact_num_states is not None and len(counter) != exact_num_states: continue

            # --- Iterate through Bit Flip Combinations ---
            found_match_for_this_entry = False
            for bit_positions_tuple in all_combinations:
                if found_match_for_this_entry: break

                bit_positions = list(bit_positions_tuple)

                try:
                    # Apply flips and create mapping
                    flipped_targets = [apply_bit_flip(state, bit_positions) for state in local_target_states]
                    # Map original state to its flipped version for coefficient lookup
                    orig_to_flipped = {orig: flipped for orig, flipped in zip(local_target_states, flipped_targets)}

                    # Condition 1: All flipped states must exist
                    all_states_exist = all(flipped_state in counter for flipped_state in flipped_targets)

                    if all_states_exist:
                        # Condition 2: Check coefficients if required
                        coeffs_match = True # Default to True if no check needed
                        if valid_target_coeffs is not None: # Check required
                            coeffs_match = False # Assume False until proven True
                            all_required_coeffs_match_so_far = True
                            for orig_state, required_coeff in valid_target_coeffs.items():
                                # Find the corresponding flipped state
                                # Should exist because all_states_exist is True and validation passed
                                flipped_state = orig_to_flipped[orig_state]
                                # Compare actual coefficient with required coefficient
                                if counter.get(flipped_state, 0) != required_coeff:
                                    all_required_coeffs_match_so_far = False
                                    break # Mismatch found, no need to check others
                            if all_required_coeffs_match_so_far:
                                coeffs_match = True # All specified coefficients matched

                        # If both conditions met (state existence and, if required, coeff match)
                        if coeffs_match:
                            # Success! Record the result
                            found_flipped_states_with_coeffs = {ft: counter[ft] for ft in flipped_targets}

                            result_tuple = (
                                key, graph_obj, graph_idx, perfect_matching_data,
                                counter, bit_positions, found_flipped_states_with_coeffs
                            )
                            results.append(result_tuple)
                            found_match_for_this_entry = True # Set flag
                            break # Stop checking other flips for this Counter entry

                except Exception as e:
                    print(f"Warning: Unexpected error during processing for key '{key}', item {i}, flips {bit_positions}: {e}. Skipping combo.", file=sys.stderr)
                    continue # Try next flip combination

    return results


# def filter_by_state_count(result_dict, min_states=None, max_states=None, exact_states=None, hash_key=None):
#     """
#     Filter results to include only those with a specific number of quantum states.
    
#     Parameters:
#     -----------
#     result_dict : dict
#         Result dictionary from process_graph_dict()
#     min_states : int, optional
#         Minimum number of states required (default: None)
#     max_states : int, optional
#         Maximum number of states allowed (default: None)
#     exact_states : int, optional
#         Exact number of states required (default: None)
#     hash_key : str, optional
#         Specific hash key to filter within (default: None, filter all hashes)
        
#     Returns:
#     --------
#     dict
#         Filtered results dictionary with the same structure as the input
#     """
#     filtered_results = {}
    
#     # Determine which hash keys to process
#     if hash_key is not None:
#         if hash_key not in result_dict:
#             return {}
#         hash_keys = [hash_key]
#     else:
#         hash_keys = result_dict.keys()
    
#     # Process each hash key
#     for key in hash_keys:
#         filtered_list = []
        
#         for state_data in result_dict[key]:
#             counter = state_data[0]  # State coefficient Counter
#             num_states = len(counter)
            
#             # Check if number of states meets the criteria
#             matches = True
#             if exact_states is not None:
#                 matches = (num_states == exact_states)
#             else:
#                 if min_states is not None and num_states < min_states:
#                     matches = False
#                 if max_states is not None and num_states > max_states:
#                     matches = False
            
#             # Add to filtered results if it matches
#             if matches:
#                 filtered_list.append(state_data)
        
#         # Only add to results if there are filtered items
#         if filtered_list:
#             filtered_results[key] = filtered_list
    
#     return filtered_results

# def get_all_quantum_states(
#     result_dict: Dict[str, List[List[Any]]],
#     hash_key: Optional[str] = None
#     ) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
#     """
#     Extracts all unique quantum states (bit strings) found in the results,
#     along with their coefficients and the original graph index where they appeared.

#     Parameters:
#     -----------
#     result_dict : Dict[str, List[List[Any]]]
#         The dictionary of processed results.
#     hash_key : Optional[str], optional
#         If provided, extract states only from this hash group.

#     Returns:
#     --------
#     Dict[str, Dict[str, List[Tuple[int, int]]]]
#         A dictionary mapping:
#         hash_key -> { state_string: [(coefficient, graph_original_index), ...], ... }
#     """
#     all_states_found: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))

#     # Determine which hash keys to process
#     search_keys = []
#     if hash_key is not None:
#         if hash_key in result_dict:
#             search_keys = [hash_key]
#         else:
#             print(f"Warning: Specified hash_key '{hash_key}' not found.")
#             return {}
#     else:
#         search_keys = list(result_dict.keys())

#     # Iterate through the selected groups and entries
#     for key in search_keys:
#         for state_data_entry in result_dict.get(key, []):
#             counter = state_data_entry[0]
#             original_graph_index = state_data_entry[3]

#             if not isinstance(counter, Counter): continue

#             # Add each state and its info to the result dictionary
#             for state, coefficient in counter.items():
#                 all_states_found[key][state].append((coefficient, original_graph_index))

#     # Convert defaultdicts back to regular dicts for the final output
#     final_dict = {k: dict(v) for k, v in all_states_found.items()}
#     return final_dict


# # ==============================================================================
# # Section 4: Quantum State Equivalence Checking and Filtering Functions
# # ==============================================================================

# # --- Permutation-based Equivalence ---

# def apply_permutation_to_keys(counter: CounterType[str], permutation: Sequence[int]) -> CounterType[str]:
#     """
#     Applies a permutation to the keys (bit strings) of a Counter.

#     Example: counter={'010': 5}, permutation=(1, 0, 2) -> returns {'100': 5}

#     Parameters:
#     -----------
#     counter : CounterType[str]
#         The input Counter where keys are bit strings of the same length.
#     permutation : Sequence[int]
#         A sequence (e.g., tuple) representing the new order of indices.
#         Example: (1, 0, 2) means the bit at index 1 comes first, then index 0, then index 2.

#     Returns:
#     --------
#     CounterType[str]
#         A new Counter with permuted keys.

#     Raises:
#     ------
#     ValueError:
#         If the permutation length doesn't match the key length or if keys have varying lengths.
#     IndexError:
#         If permutation contains invalid indices.
#     """
#     transformed_counter: CounterType[str] = Counter()
#     key_length = -1

#     if not counter:
#         return transformed_counter # Return empty counter if input is empty

#     for key, value in counter.items():
#         if key_length == -1:
#             key_length = len(key)
#             # Check permutation validity once
#             if len(permutation) != key_length:
#                 raise ValueError(f"Permutation length {len(permutation)} must match key length {key_length}.")
#             if not all(0 <= i < key_length for i in permutation):
#                  raise IndexError("Permutation contains invalid indices.")
#             if len(set(permutation)) != key_length:
#                  raise ValueError("Permutation must contain unique indices.")

#         elif len(key) != key_length:
#             raise ValueError("All keys in the Counter must have the same length.")

#         # Apply permutation
#         try:
#             new_key_list = [key[i] for i in permutation]
#             new_key = ''.join(new_key_list)
#             transformed_counter[new_key] = value
#         except IndexError:
#             # This should theoretically not happen if initial checks pass
#             raise IndexError(f"Error applying permutation {permutation} to key {key}. Invalid index.")

#     return transformed_counter


# def remove_duplicate_counters(data_list: List[List[Any]]) -> List[List[Any]]:
#     """
#     Removes duplicate entries from a list based on permutation equivalence of their state Counters.

#     Compares each Counter with others, applying all possible permutations to the keys.
#     Keeps only one representative for each permutation equivalence class.

#     Parameters:
#     -----------
#     data_list : List[List[Any]]
#         A list where each inner list is expected to have a Counter object
#         at index 0 (e.g., [Counter, matchings, graph, index]).

#     Returns:
#     --------
#     List[List[Any]]
#         A filtered list containing only entries with unique Counters up to key permutation.
#     """
#     if not data_list:
#         return []

#     indices_to_remove: Set[int] = set()
#     n_elements = len(data_list)
#     unique_representatives: List[List[Any]] = [] # Stores the first encountered representative

#     # Pre-calculate permutations if all counters have the same key length
#     first_counter = data_list[0][0]
#     if not isinstance(first_counter, Counter) or not first_counter:
#          print("Warning: First element's counter is invalid or empty in remove_duplicate_counters. Cannot determine permutations.")
#          # Fallback or error handling needed - let's assume valid for now
#          # Or iterate and determine per counter comparison if lengths vary (less efficient)
#          return data_list # Return original if cannot proceed safely

#     try:
#         n_bits = len(next(iter(first_counter))) # Get length from first key of first counter
#         all_permutations = list(itertools.permutations(range(n_bits)))
#     except (StopIteration, TypeError):
#         print("Warning: Could not determine key length from first counter. Skipping permutation check.")
#         return data_list # Return original if cannot proceed safely


#     for i in range(n_elements):
#         if i in indices_to_remove:
#             continue # Skip if already marked for removal

#         # Add the current element as a potential unique representative
#         unique_representatives.append(data_list[i])
#         counter1 = data_list[i][0]
#         if not isinstance(counter1, Counter): continue # Skip invalid entries

#         # Check against subsequent elements
#         for j in range(i + 1, n_elements):
#             if j in indices_to_remove:
#                 continue

#             counter2 = data_list[j][0]
#             if not isinstance(counter2, Counter): continue

#             # Basic check: must have same number of states and same coefficients (values)
#             if len(counter1) != len(counter2) or sorted(counter1.values()) != sorted(counter2.values()):
#                 continue

#             # Check if counter2 is a permutation of counter1
#             is_permutation_equivalent = False
#             try:
#                 # Optimization: Check if key lengths match before permutations
#                 len1 = len(next(iter(counter1))) if counter1 else -1
#                 len2 = len(next(iter(counter2))) if counter2 else -1
#                 if len1 != len2 or len1 == -1:
#                      continue # Cannot be permutation equivalent if lengths differ or empty

#                 # Check against all permutations of counter1
#                 for p in all_permutations:
#                     transformed_counter1 = apply_permutation_to_keys(counter1, p)
#                     if transformed_counter1 == counter2:
#                         is_permutation_equivalent = True
#                         break # Found equivalence
#             except Exception as e:
#                  print(f"Warning: Error during permutation check between index {i} and {j}: {e}. Assuming not equivalent.")

#             if is_permutation_equivalent:
#                 indices_to_remove.add(j) # Mark j for removal

#     # Filter the original list based on indices not marked for removal
#     final_unique_list = [data_list[k] for k in range(n_elements) if k not in indices_to_remove]

#     print(f"Removed {len(indices_to_remove)} entries based on key permutation equivalence. Kept {len(final_unique_list)}.")
#     return final_unique_list


# def remove_duplicate_counters_full_list(grouped_counters: Dict[Any, List[List[Any]]]) -> Dict[Any, List[List[Any]]]:
#     """
#     Applies `remove_duplicate_counters` to each list within a dictionary of grouped results.

#     Parameters:
#     -----------
#     grouped_counters : Dict[Any, List[List[Any]]]
#          A dictionary where keys might represent groups (e.g., by coefficient counts)
#          and values are lists of [Counter, ...] entries.

#     Returns:
#     --------
#     Dict[Any, List[List[Any]]]
#         A dictionary with the same keys but with filtered lists as values.
#     """
#     save_filtered_data: Dict[Any, List[List[Any]]] = {}
#     print("Applying permutation-based duplicate removal to grouped counters...")
#     for key, data_list in grouped_counters.items():
#         # print(f"Processing group with key: {key} (Size: {len(data_list)})")
#         filtered_list = remove_duplicate_counters(data_list)
#         save_filtered_data[key] = filtered_list
#     print("Permutation-based filtering complete for all groups.")
#     return save_filtered_data

# # --- Bit-Flip-based Equivalence ---


# def flip_multiple_bits(binary_str: str, positions: Iterable[int]) -> str:
#     """
#     Flips bits at multiple specified positions in a binary string.

#     Parameters:
#     -----------
#     binary_str : str
#         The input binary string.
#     positions : Iterable[int]
#         An iterable (e.g., list, tuple) of 0-indexed positions to flip.

#     Returns:
#     --------
#     str
#         The binary string with bits flipped at the specified positions.
#     """
#     bit_list = list(binary_str)
#     str_len = len(binary_str)
#     for pos in positions:
#         if not 0 <= pos < str_len:
#             raise IndexError(f"Position {pos} is out of bounds for string of length {str_len}.")
#         if bit_list[pos] == '0':
#             bit_list[pos] = '1'
#         elif bit_list[pos] == '1':
#             bit_list[pos] = '0'
#         else:
#             raise ValueError(f"Character at index {pos} is '{bit_list[pos]}', not '0' or '1'.")
#     return ''.join(bit_list)


# def find_transformation(counter1: CounterType[str], counter2: CounterType[str]) -> Optional[Tuple[int, ...]]:
#     """
#     Checks if counter2 can be obtained from counter1 by flipping the same set of bits
#     in all keys of counter1.

#     Parameters:
#     -----------
#     counter1 : CounterType[str]
#         The starting Counter.
#     counter2 : CounterType[str]
#         The target Counter.

#     Returns:
#     --------
#     Optional[Tuple[int, ...]]
#         A tuple containing the 0-indexed positions of the bits that need to be flipped
#         to transform counter1 to counter2. Returns None if no such transformation exists
#         or if counters are incompatible (different sizes, different values, empty, etc.).
#     """
#     if len(counter1) != len(counter2) or not counter1:
#         return None # Cannot transform if sizes differ or empty
#     if sorted(counter1.values()) != sorted(counter2.values()):
#         return None # Coefficients must match

#     try:
#         bit_length = len(next(iter(counter1)))
#         if not all(len(k) == bit_length for k in counter1):
#              print("Warning: Keys in counter1 have inconsistent lengths.")
#              return None
#         if not all(len(k) == bit_length for k in counter2):
#              print("Warning: Keys in counter2 have inconsistent lengths.")
#              return None

#     except (StopIteration, TypeError):
#         return None # Handle empty counter or non-string keys

#     # Try flipping combinations of bit positions
#     # r=0 corresponds to no flips (identity transformation)
#     for r in range(bit_length + 1):
#         for positions in itertools.combinations(range(bit_length), r):
#             try:
#                 # Apply the same bit flip to all keys in counter1
#                 flipped_data = Counter({flip_multiple_bits(bstr, positions): val for bstr, val in counter1.items()})

#                 # Check if the flipped counter matches counter2
#                 if flipped_data == counter2:
#                     return positions # Return the positions that worked
#             except (IndexError, ValueError) as e:
#                  # Should not happen if length checks passed, but handle defensively
#                  print(f"Error during bit flip check with positions {positions}: {e}")
#                  continue # Try next combination

#     return None # No transformation found


# def check_transformations(data_list: List[List[Any]]) -> Tuple[List[Tuple[int, int, Tuple[int, ...]]], List[List[Any]]]:
#     """
#     Checks for bit-flip equivalence between pairs of Counters in a list.

#     Identifies pairs (i, j) where data_list[j][0] can be obtained from data_list[i][0]
#     by applying the same bit flips. Returns the transformations found and a list
#     of representatives that are not transformable into each other via bit flips.

#     Parameters:
#     -----------
#     data_list : List[List[Any]]
#         List of [Counter, ...] entries.

#     Returns:
#     --------
#     Tuple[List[Tuple[int, int, Tuple[int, ...]]], List[List[Any]]]
#         - List of transformations found: (index_i, index_j, flip_positions_tuple)
#         - List of non-transformable representatives (one from each equivalence class).
#     """
#     transformations: List[Tuple[int, int, Tuple[int, ...]]] = []
#     # Keep track of indices that are equivalent to an earlier index
#     equivalent_indices: Set[int] = set()
#     n_elements = len(data_list)

#     if n_elements < 2:
#         return [], data_list # No transformations possible with less than 2 elements

#     # Extract Counters for easier access
#     counters: List[Optional[CounterType[str]]] = []
#     for item in data_list:
#         if isinstance(item[0], Counter):
#             counters.append(item[0])
#         else:
#             counters.append(None) # Mark invalid entries

#     for i in range(n_elements):
#         if i in equivalent_indices or counters[i] is None:
#             continue # Skip if already marked as equivalent or invalid

#         for j in range(i + 1, n_elements):
#             if j in equivalent_indices or counters[j] is None:
#                 continue

#             # Check if counter j can be transformed from counter i
#             bit_positions = find_transformation(counters[i], counters[j])
#             if bit_positions is not None:
#                 transformations.append((i, j, bit_positions))
#                 equivalent_indices.add(j) # Mark j as equivalent to i

#             # Also check the reverse transformation (i from j) - might be redundant if find_transformation is symmetric
#             # bit_positions_rev = find_transformation(counters[j], counters[i])
#             # if bit_positions_rev is not None:
#             #     # Found transformation, ensure j isn't already marked relative to something else
#             #     if j not in equivalent_indices:
#             #          transformations.append((j, i, bit_positions_rev)) # Record j->i
#             #          equivalent_indices.add(i) # Mark i as equivalent to j (if not already marked)
#             #     # Avoid marking both i and j if they are equivalent to each other


#     # Collect the representatives (those not marked as equivalent to an earlier one)
#     non_transformable_representatives = [data_list[k] for k in range(n_elements) if k not in equivalent_indices]

#     print(f"Found {len(transformations)} bit-flip transformations. Kept {len(non_transformable_representatives)} representatives.")
#     return transformations, non_transformable_representatives


# def check_tranf_full_data(grouped_data: Dict[Any, List[List[Any]]]) -> Dict[Any, List[List[Any]]]:
#     """
#     Applies `check_transformations` (bit-flip equivalence check) to each list
#     within a dictionary of grouped results.

#     Parameters:
#     -----------
#     grouped_data : Dict[Any, List[List[Any]]]
#          Dictionary where values are lists of [Counter, ...] entries.

#     Returns:
#     --------
#     Dict[Any, List[List[Any]]]
#         Dictionary with the same keys but values replaced by the list of
#         non-transformable representatives for each group.
#     """
#     final_representatives_dict: Dict[Any, List[List[Any]]] = {}
#     print("Applying bit-flip-based duplicate removal to grouped data...")
#     for key, data_list in grouped_data.items():
#         # print(f"Processing group {key} for bit-flip equivalence (Size: {len(data_list)})")
#         _, non_transformable_representatives = check_transformations(data_list)
#         final_representatives_dict[key] = non_transformable_representatives
#     print("Bit-flip-based filtering complete for all groups.")
#     return final_representatives_dict


# # --- Hadamard-based Equivalence ---

# def counter_to_state_vector(counter: CounterType[str], n_qubits: int) -> np.ndarray:
#     """
#     Converts a state Counter into a normalized NumPy state vector.

#     Assumes keys are binary strings representing computational basis states.
#     Coefficients in the Counter are treated as *counts* or *proportions*
#     and are converted to amplitudes (sqrt(probability)).

#     Parameters:
#     -----------
#     counter : CounterType[str]
#         Counter where keys are binary strings of length n_qubits, and values
#         are positive numbers representing counts or relative frequencies.
#     n_qubits : int
#         The number of qubits (length of the binary strings).

#     Returns:
#     --------
#     np.ndarray
#         A complex NumPy array of shape (2**n_qubits,) representing the
#         normalized quantum state vector.

#     Raises:
#     ------
#     ValueError: If keys have incorrect length, values are non-positive,
#                 or n_qubits is non-positive.
#     """
#     if n_qubits <= 0:
#         raise ValueError("n_qubits must be positive.")
#     if not counter:
#         # Return zero vector if counter is empty
#         return np.zeros(2**n_qubits, dtype=complex)

#     total_counts = sum(counter.values())
#     if total_counts <= 0:
#         # Check if all values are non-positive
#         if all(v <= 0 for v in counter.values()):
#              print("Warning: All counts in counter are non-positive. Returning zero vector.")
#              return np.zeros(2**n_qubits, dtype=complex)
#         else:
#              raise ValueError("Sum of counts must be positive for normalization.")

#     state_vector = np.zeros(2**n_qubits, dtype=complex)

#     for bitstring, count in counter.items():
#         if len(bitstring) != n_qubits:
#             raise ValueError(f"Key '{bitstring}' has length {len(bitstring)}, expected {n_qubits}.")
#         if count < 0:
#             raise ValueError(f"Count for state '{bitstring}' is negative ({count}). Counts must be non-negative.")
#         if count == 0:
#              continue # Skip states with zero count

#         try:
#             index = int(bitstring, 2) # Convert binary string to integer index
#         except ValueError:
#             raise ValueError(f"Key '{bitstring}' is not a valid binary string.")

#         # Amplitude is sqrt(probability) = sqrt(count / total_counts)
#         amplitude = np.sqrt(count / total_counts)
#         state_vector[index] = amplitude # Assign amplitude (real in this case)

#     # Optional: Verify normalization (sum of squared amplitudes should be ~1)
#     # norm_sq = np.sum(np.abs(state_vector)**2)
#     # if not np.isclose(norm_sq, 1.0):
#     #     print(f"Warning: State vector normalization failed. Sum of squares = {norm_sq}")

#     return state_vector


# def apply_hadamard_to_qubits(n_qubits: int, hadamard_qubits: Sequence[int]) -> np.ndarray:
#     """
#     Constructs the matrix operator for applying Hadamard gates to a subset of qubits.

#     Creates a (2**n_qubits x 2**n_qubits) matrix representing H applied to
#     specified qubits and Identity (I) applied to others, using Kronecker products.

#     Parameters:
#     -----------
#     n_qubits : int
#         Total number of qubits in the system.
#     hadamard_qubits : Sequence[int]
#         A sequence (list, tuple) of 0-indexed qubit indices to apply Hadamard to.

#     Returns:
#     --------
#     np.ndarray
#         The (2**n_qubits, 2**n_qubits) complex NumPy array for the operator.

#     Raises:
#     ------
#     IndexError: If any index in hadamard_qubits is out of bounds.
#     """
#     if n_qubits <= 0:
#         raise ValueError("n_qubits must be positive.")

#     # Single qubit Hadamard gate
#     H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
#     # Single qubit Identity gate
#     I = np.eye(2, dtype=complex)

#     # Build the operator using Kronecker product
#     operator = np.array([[1.0]], dtype=complex) # Start with a 1x1 identity for kronecker product

#     for i in range(n_qubits):
#         if not 0 <= i < n_qubits:
#              # This check is redundant given the loop range, but good practice
#              raise IndexError(f"Internal error: Qubit index {i} out of bounds.")

#         # Choose H or I for the i-th qubit
#         single_qubit_op = H if i in hadamard_qubits else I

#         # Apply Kronecker product
#         # np.kron(A, B) reverses the standard tensor product order if thinking qubit 0 first
#         # To match standard convention (e.g., Qiskit), build from right-to-left or reverse indices.
#         # Let's stick to left-to-right for consistency with loop, assuming qubit 0 is leftmost.
#         operator = np.kron(operator, single_qubit_op)

#     return operator


# def get_unique_transformed_states_from_dict_v2(
#     data_dict: Dict[Any, List[List[Any]]]
#     ) -> Dict[Any, List[List[Any]]]:
#     """
#     Filters dictionary of state data, keeping only one representative for states
#     that are equivalent under local Hadamard transformations applied to subsets of qubits.

#     Compares state vectors derived from Counters.

#     Parameters:
#     -----------
#     data_dict : Dict[Any, List[List[Any]]]
#         Dictionary where values are lists of [Counter, ...] entries.

#     Returns:
#     --------
#     Dict[Any, List[List[Any]]]
#         Dictionary with the same keys but values filtered to keep only unique
#         representatives under local Hadamard equivalence.
#     """
#     print("Applying Hadamard-based equivalence filtering...")
#     all_original_states: List[Tuple[Any, int, np.ndarray, List[Any]]] = [] # (key, original_index, state_vector, original_data_entry)
#     key_order_map: Dict[Tuple[Any, int], int] = {} # Maps (key, original_index) to index in all_original_states

#     # 1. Convert all counters to state vectors and store them
#     current_global_index = 0
#     for key, data_list in data_dict.items():
#         for idx, entry in enumerate(data_list):
#             counter = entry[0]
#             if not isinstance(counter, Counter) or not counter:
#                 print(f"Warning: Invalid counter for key '{key}', index {idx}. Skipping.")
#                 continue
#             try:
#                 n_qubits = len(next(iter(counter)))
#                 state_vector = counter_to_state_vector(counter, n_qubits)
#                 all_original_states.append((key, idx, state_vector, entry))
#                 key_order_map[(key, idx)] = current_global_index
#                 current_global_index += 1
#             except (StopIteration, ValueError, TypeError) as e:
#                 print(f"Error converting counter to state vector for key '{key}', index {idx}: {e}. Skipping.")

#     n_total_states = len(all_original_states)
#     if n_total_states < 2:
#         print("Less than 2 valid states found, no Hadamard comparison needed.")
#         return data_dict # Return original if nothing to compare

#     # 2. Precompute all possible Hadamard transformations for each state
#     # Stores {global_index: [transformed_vector1, transformed_vector2, ...]}
#     transformed_state_cache: Dict[int, List[np.ndarray]] = defaultdict(list)
#     print(f"Precomputing Hadamard transformations for {n_total_states} states...")
#     for i in range(n_total_states):
#         _, _, state_vector_i, entry_i = all_original_states[i]
#         counter_i = entry_i[0]
#         try:
#              n_qubits_i = len(next(iter(counter_i)))
#              # Generate transformations by applying H to all non-empty subsets of qubits
#              transformed_state_cache[i].append(state_vector_i) # Include original vector
#              for k in range(1, n_qubits_i + 1):
#                  for qubit_indices in itertools.combinations(range(n_qubits_i), k):
#                      H_op = apply_hadamard_to_qubits(n_qubits_i, qubit_indices)
#                      transformed_vector = H_op @ state_vector_i # Apply operator
#                      transformed_state_cache[i].append(transformed_vector)
#         except Exception as e:
#              print(f"Error generating transformations for state index {i}: {e}")
#              # Cache will be incomplete for this index


#     # 3. Compare states: Check if state j is equivalent to any transformation of state i (where i < j)
#     equivalent_indices: Set[int] = set() # Stores global indices to remove
#     print("Comparing states for Hadamard equivalence...")
#     for i in range(n_total_states):
#         if i in equivalent_indices:
#             continue

#         # Get transformations of state i (if computed successfully)
#         transforms_i = transformed_state_cache.get(i, [])
#         if not transforms_i: continue # Skip if no transformations available

#         for j in range(i + 1, n_total_states):
#             if j in equivalent_indices:
#                 continue

#             # Get original state vector j
#             _, _, state_vector_j, _ = all_original_states[j]

#             # Check if state_vector_j is close to any vector in transforms_i
#             is_equivalent = False
#             for transformed_i in transforms_i:
#                 # Use np.allclose for comparing floating-point vectors
#                 # Need to consider global phase: check if |<psi|phi>|^2 is close to 1
#                 try:
#                      # Ensure vectors have the same shape
#                      if state_vector_j.shape == transformed_i.shape:
#                           inner_product = np.vdot(transformed_i, state_vector_j) # <transformed_i | state_j>
#                           overlap_sq = np.abs(inner_product)**2
#                           # Check if overlap squared is close to 1 (vectors are the same up to global phase)
#                           if np.allclose(overlap_sq, 1.0, atol=1e-8): # Adjust tolerance as needed
#                                is_equivalent = True
#                                break
#                 except Exception as e_comp:
#                      print(f"Error comparing state {i} transformation with state {j}: {e_comp}")


#             if is_equivalent:
#                 equivalent_indices.add(j) # Mark state j as equivalent to state i

#     # 4. Reconstruct the dictionary with only unique representatives
#     final_unique_data_dict: Dict[Any, List[List[Any]]] = defaultdict(list)
#     kept_count = 0
#     for i in range(n_total_states):
#         if i not in equivalent_indices:
#             key, _, _, original_entry = all_original_states[i]
#             final_unique_data_dict[key].append(original_entry)
#             kept_count += 1

#     print(f"Hadamard filtering complete. Kept {kept_count} unique representatives out of {n_total_states} initial states.")
#     # Convert defaultdict back to dict if necessary
#     return dict(final_unique_data_dict)


########################################################


def get_bipartite_sets(G: ig.Graph) -> Tuple[List[int], List[int]]:
    """Extracts the two node sets (partitions) of a bipartite graph."""
    # (이전 버전의 get_bipartite_sets 함수 코드를 여기에 붙여넣으세요)
    U: List[int] = []
    V: List[int] = []
    if 'bipartite' in G.vs.attributes():
        try:
            U = [v.index for v in G.vs if v['bipartite'] == 0]
            V = [v.index for v in G.vs if v['bipartite'] == 1]
        except KeyError:
            raise ValueError("Graph has 'bipartite' attribute but some nodes lack the value.")
    elif 'category' in G.vs.attributes():
        try:
            U = [v.index for v in G.vs if v['category'] in ['system_nodes', 'ancilla_nodes']]
            V = [v.index for v in G.vs if v['category'] == 'sculpting_nodes']
        except KeyError:
            raise ValueError("Graph has 'category' attribute but some nodes lack the value.")
    else:
        raise ValueError("Cannot determine bipartite sets. Graph lacks 'bipartite' or 'category' node attributes.")
    if not U or not V:
         print(f"Warning: One or both bipartite sets are empty (U: {len(U)}, V: {len(V)}).")
    if len(U) + len(V) != G.vcount():
         raise ValueError("Sum of nodes in bipartite sets does not match total vertex count.")
    return U, V

# --- 1. 완전 매칭 구조 찾기 함수 ---
def _dfs_pm_structures(
    G: ig.Graph,
    U: List[int],
    V: List[int],
    u_index: int,
    current_matching_edges: List[Tuple[int, int]], # Store (u, v) tuples
    matched_V: Dict[int, bool],
    all_pm_structures: List[List[Tuple[int, int]]]
) -> None:
    """Helper DFS to find perfect matching structures (edge lists)."""
    if u_index == len(U):
        all_pm_structures.append(current_matching_edges[:])
        return

    if u_index >= len(U): return # Should not happen with correct base case
    u = U[u_index]

    # Use neighbors for potentially better performance if graph is sparse
    # neighbors_of_u = G.neighbors(u) # Or iterate through V as before
    for v in V: # Or neighbors_of_u if using that optimization
        # Check if edge exists (redundant if using G.neighbors, needed if iterating V)
        if G.get_eid(u, v, directed=False, error=False) == -1:
             continue

        if not matched_V.get(v, False):
            current_matching_edges.append((u, v))
            matched_V[v] = True
            _dfs_pm_structures(G, U, V, u_index + 1, current_matching_edges, matched_V, all_pm_structures)
            matched_V[v] = False
            current_matching_edges.pop()

def find_all_perfect_matching_structures(G: ig.Graph, U: List[int], V: List[int]) -> List[List[Tuple[int, int]]]:
    """Finds all perfect matching structures (lists of edges) in G."""
    if len(U) != len(V) or not U:
        return []

    all_pm_structures: List[List[Tuple[int, int]]] = []
    matched_V: Dict[int, bool] = {v_node: False for v_node in V}
    _dfs_pm_structures(G, U, V, 0, [], matched_V, all_pm_structures)
    return all_pm_structures

# --- 2. 고유 간선 식별 (calculate 함수 내부에 통합 가능) ---
# Helper integrated into calculate function below

def calculate_all_signed_states(
    G: ig.Graph,
    U: List[int],
    V: List[int],
    list_of_pms: List[List[Tuple[int, int]]]
# ===> Return type hint changed <===
) -> Tuple[List[Tuple[int, int]], List[Tuple[Tuple[int, ...], CounterType[str]]], List[Tuple[List[Tuple[int, int]], str]]]:
    """
    Calculates all 2^K final state vectors based on global sign assignments.
    Now also returns a mapping from PM structures to their base state strings.

    Args:
        G: The igraph.Graph object. Assumed to have original weights (1.0, 2.0, 3.0).
        U: List of nodes in partition U.
        V: List of nodes in partition V.
        list_of_pms: List of perfect matchings, where each PM is a list of (u, v) edge tuples.

    Returns:
        Tuple containing:
        - ordered_unique_edges: List of unique edges (u,v) defining the sign tuple order.
        - results_list: List of tuples, each being (global_sign_tuple, state_counter).
        - pm_to_base_state_map_list: List of tuples, each being (pm_structure_edges, base_state_string).
        Returns ([], [], []) on error or if no PMs.
    """
    if not list_of_pms:
        # ===> Return value structure changed <===
        return [], [], []

    # --- Identify unique edges and prepare weights (same as before) ---
    unique_edges_set: Set[Tuple[int, int]] = set()
    for pm in list_of_pms:
        for u, v in pm:
            unique_edges_set.add(tuple(sorted((u, v))))
    ordered_unique_edges = list(unique_edges_set) # Assign fixed order
    K = len(ordered_unique_edges)
    edge_to_index: Dict[Tuple[int, int], int] = {edge: idx for idx, edge in enumerate(ordered_unique_edges)}
    print(f"DEBUG: Found {len(list_of_pms)} PM structures involving {K} unique edges.")
    if K > 25: print(f"WARNING: High number of unique edges ({K}). Calculation might be very slow (2^{K} combinations).")

    original_weights: Dict[Tuple[int, int], float] = {}
    valid_weights = True
    for u_edge, v_edge in ordered_unique_edges:
        w = get_edge_weight(G, u_edge, v_edge)
        if w is None:
            print(f"ERROR: Weight not found for edge ({u_edge},{v_edge}) in E_PM. Cannot proceed.")
            valid_weights = False; break
        original_weights[(u_edge, v_edge)] = w
    if not valid_weights: return [], [], []

    u_indices_map = {node_idx: i for i, node_idx in enumerate(U)}
    results_list: List[Tuple[Tuple[int, ...], CounterType[str]]] = [] # Stores (sign_tuple, state_counter)
    # ===> Initialize PM structure-to-state string mapping list <===
    pm_to_base_state_map_list: List[Tuple[List[Tuple[int, int]], str]] = []

    # --- Calculate base state string first for each PM structure ---
    # (Done outside the 2^K loop for efficiency)
    pm_base_states = {} # Temporary dictionary for storage (PM tuple -> state string)
    for pm_idx, pm_edges in enumerate(list_of_pms):
        base_state_string_parts = []
        current_pm_edge_list_for_map = [] # PM structure for storing the mapping (list of tuples)
        for u, v in pm_edges:
            current_pm_edge_list_for_map.append((u, v)) # Store structure for mapping
            edge_tuple_sorted = tuple(sorted((u, v)))
            # Ensure edge exists in original_weights (should, due to construction)
            original_weight = original_weights.get(edge_tuple_sorted)
            if original_weight is None:
                 print(f"Internal Error: Weight lookup failed for {edge_tuple_sorted}. Skipping PM for base state calc.")
                 # Or handle differently
                 continue # Skip this edge for base state calculation

            state_char = None
            if isclose(original_weight, 1.0): state_char = '0'
            elif isclose(original_weight, 2.0): state_char = '1'
            # Ignore 3.0 etc.

            if state_char is not None:
                u_node_in_edge = u if u in u_indices_map else v
                sort_key = u_indices_map.get(u_node_in_edge, float('inf'))
                base_state_string_parts.append((sort_key, state_char))

        base_state_string_parts.sort()
        base_state_string = ''.join([char for _, char in base_state_string_parts])
        # Store the calculated base state string along with the PM structure
        pm_key = tuple(sorted(pm_edges)) # Key to identify the PM structure (e.g., sorted tuple)
        pm_base_states[pm_key] = base_state_string
        # Also add to the list for final return
        pm_to_base_state_map_list.append((current_pm_edge_list_for_map, base_state_string))


    # --- Loop through 2^K global sign combinations ---
    for global_sign_tuple in itertools.product([1, -1], repeat=K):
        current_state_vector = defaultdict(int)
        # --- Iterate through all PM structures ---
        for pm_idx, pm_edges in enumerate(list_of_pms): # pm_edges is List[Tuple[int, int]]
            pm_overall_sign = 1
            # --- Calculate sign for each PM ---
            for u, v in pm_edges:
                edge_tuple_sorted = tuple(sorted((u, v)))
                # Weight should exist based on pre-fetch
                original_weight = original_weights[edge_tuple_sorted]
                # Index should exist based on construction
                edge_idx = edge_to_index[edge_tuple_sorted]
                sigma = global_sign_tuple[edge_idx]
                effective_weight = sigma * original_weight
                # Determine edge sign contribution (NEW RULE: negative effective weight contributes -1)
                edge_sign = -1 if effective_weight < 0 else 1 # <<< 여기가 수정된 부분 (Use the latest rule)
                pm_overall_sign *= edge_sign

            # --- Use pre-calculated base state string and aggregate ---
            pm_key = tuple(sorted(pm_edges)) # Same key used above
            # Base state should exist based on pre-calculation
            base_state_string_for_pm = pm_base_states[pm_key] # Look up pre-calculated value
            current_state_vector[base_state_string_for_pm] += pm_overall_sign

        # --- Process final state vector for the current sign combination ---
        final_coeffs_for_this_state = {s: c for s, c in current_state_vector.items() if c != 0}
        # (Optional GCD normalization logic - same as before) ...
        if final_coeffs_for_this_state:
             coeffs_abs = [abs(c) for c in final_coeffs_for_this_state.values()]
             if len(coeffs_abs) >= 2:
                 try:
                     common_divisor = reduce(gcd, coeffs_abs)
                     if common_divisor > 1:
                         final_coeffs_for_this_state = {s: c // common_divisor for s, c in final_coeffs_for_this_state.items()}
                 except Exception as e_gcd:
                     print(f"Warning: GCD normalization failed for one state: {e_gcd}")
        # ...
        state_counter = Counter(final_coeffs_for_this_state)
        # Add global_sign_tuple when storing result
        results_list.append((global_sign_tuple, state_counter))

    # Optional: Deduplication logic for results_list if needed
    # ...

    # ===> Return value structure changed: return 3 items <===
    return ordered_unique_edges, results_list, pm_to_base_state_map_list

# --- 4. 메인 제어 함수 (find_pm_of_bigraph 수정 예시) ---
def find_pm_results_final(graph_list: List[ig.Graph]) -> List[List[Any]]:
    """
    Processes graphs to find all quantum states based on global edge sign combinations.
    Returns a list where each item corresponds to an input graph and contains
    a LIST of final state Counters (one for each global sign combination).
    """
    processed_results = []
    for i, G in enumerate(graph_list):
        print(f"\nProcessing graph index {i}...")
        try:
            U, V = get_bipartite_sets(G)
            if not U or not V or len(U) != len(V):
                 print(f"Skipping graph {i}: Invalid partitions (U:{len(U)}, V:{len(V)})")
                 continue

            # 1. Find PM Structures
            pm_structures = find_all_perfect_matching_structures(G, U, V)
            if not pm_structures:
                 print(f"Skipping graph {i}: No perfect matchings found.")
                 continue

            # Optional: Apply filters based on pm_structures count or unused edges
            # Example Filter 1: >= 2 PM structures
            if len(pm_structures) < 2:
                print(f"Skipping graph {i}: Less than 2 PM structures found ({len(pm_structures)}).")
                continue
            # Example Filter 2: Check for unused edges (needs find_unused_edges adaptation)
            # unique_pm_edges = set(tuple(sorted(e)) for pm in pm_structures for e in pm)
            # all_graph_edges = set(tuple(sorted((e.source, e.target))) for e in G.es)
            # if not unique_pm_edges.issuperset(all_graph_edges): # Or specific check
            #     print(f"Skipping graph {i}: Not all graph edges are used in PMs.")
            #     continue

            # 2. Calculate all 2^K states
            list_of_state_counters = calculate_all_signed_states(G, U, V, pm_structures)

            # 3. Store results (Graph + List of States)
            if list_of_state_counters: # Only store if calculation succeeded and potentially produced states
                 processed_results.append([
                     list_of_state_counters, # List of Counters
                     pm_structures,          # List of PM edge lists
                     G,
                     i
                 ])
            else:
                 print(f"Graph {i}: Calculation resulted in no final states (or failed).")

        except Exception as e:
            print(f"Error processing graph index {i}: {e}")
            # import traceback
            # traceback.print_exc()

    print(f"\nProcessing finished. Generated results for {len(processed_results)} graphs.")
    return processed_results


# 

# # --- perform_final_signed_analysis function (Modified for new return value) ---
# def perform_final_signed_analysis(
#     filtered_results: List[Tuple[str, Any, int, Any, CounterType[str], List[int], Dict[str, int]]]
# ) -> List[Dict[str, Any]]:
#     """
#     Performs the final analysis (Interpretation 6), now including the mapping
#     from PM structures to their base state strings in the results.
#     """
#     final_signed_analysis_results = [] # List to store the final results
#     # --- Input data validation (Optional) ---
#     if not isinstance(filtered_results, list):
#         print("Error: Expected 'filtered_results' to be a list.")
#         return [] # Return empty list
#     if not filtered_results:
#         print("Info: 'filtered_results' is empty. No graphs to process.")
#         return [] # Return empty list

#     print(f"\nStarting Stage 3: Performing 2^K analysis on {len(filtered_results)} filtered graph entries...")
#     # --- Iterate through filtered results ---
#     for entry_index, filter_entry in enumerate(filtered_results):
#         try:
#             # --- Input item structure check and data extraction ---
#             # Check return tuple structure from check_quantum_states_with_bit_flips
#             if not isinstance(filter_entry, tuple) or len(filter_entry) < 3: # Check minimum key, graph, index
#                  print(f"  Skipping entry {entry_index}: Invalid format or insufficient elements.")
#                  continue

#             hash_key = filter_entry[0]
#             graph_obj = filter_entry[1]
#             graph_idx = filter_entry[2]
#             # pm_data = filter_entry[3] # Not used directly, PM structures recalculated

#             # Validate graph_obj
#             if not isinstance(graph_obj, ig.Graph):
#                  print(f"  Skipping entry {entry_index} (Key='{hash_key}', Idx={graph_idx}): Invalid graph object type {type(graph_obj)}.")
#                  continue

#             print(f"  Processing filtered entry {entry_index}: Key='{hash_key}', Index={graph_idx}")

#             # a. Get partitions
#             U, V = get_bipartite_sets(graph_obj)
#             # Check partition validity
#             if not U or not V or len(U) != len(V):
#                  print(f"    Skipping: Invalid partitions found (U:{len(U)}, V:{len(V)})")
#                  continue

#             # b. Find perfect matching structures
#             pm_structures = find_all_perfect_matching_structures(graph_obj, U, V)
#             # Check if PM structures were found
#             if not pm_structures:
#                  print(f"    Skipping: No perfect matching structures found for this graph.")
#                  continue
#             print(f"    Found {len(pm_structures)} PM structures.")

#             # c. Calculate state vectors and PM-state mapping
#             # ===> Modified to receive 3 return values <===
#             ordered_edges, signed_states_results, pm_to_base_state = \
#                 calculate_all_signed_states(graph_obj, U, V, pm_structures)

#             # d. Store final result (including the new mapping)
#             if signed_states_results is not None: # Check calculation result (might be [] on error)
#                 final_signed_analysis_results.append({
#                     'hash_key': hash_key,
#                     'graph_index': graph_idx,
#                     'graph': graph_obj, # Decide whether to store graph object based on need
#                     # 'pm_structures': pm_structures, # Optional: Now pm_to_base_state might be more useful
#                     'ordered_unique_edges': ordered_edges, # List of unique edges defining sign tuple order
#                     'signed_states_results': signed_states_results, # List of (sign tuple, state counter)
#                     # ===> Add new mapping information <===
#                     'pm_to_base_state': pm_to_base_state # List of (pm_structure_edges, base_state_string)
#                 })
#                 print(f"    Successfully generated {len(signed_states_results)} (sign_tuple, state_vector) pairs.")
#             else:
#                 print(f"    Warning: State calculation failed or resulted in no states.")

#         except Exception as e:
#             print(f"    ERROR processing entry {entry_index} (Key='{hash_key}', Idx={graph_idx}): {e}")
#             # traceback.print_exc() # Uncomment for detailed debugging

#     print(f"\nStage 3 finished. Final analysis results generated for {len(final_signed_analysis_results)} entries.")
#     return final_signed_analysis_results

# --- Modified perform_final_signed_analysis function ---
def perform_final_signed_analysis(
    # Input type should match the return type of check_quantum_states_with_bit_flips
    filtered_results: List[Tuple[str, Any, int, Any, CounterType[str], List[int], Dict[str, int]]]
) -> List[Dict[str, Any]]:
    """
    Performs the final analysis (Interpretation 6: 2^K global sign combinations)
    on the filtered results from check_quantum_states_with_bit_flips,
    carrying forward the bit flip information and including PM-to-base-state mapping.

    Parameters:
    -----------
    filtered_results : List[Tuple[...]]
        The list returned by check_quantum_states_with_bit_flips.
        Each tuple is expected to contain at least
        (hash_key, graph_object, graph_index, ..., applied_flips_list, ...).
        Indices used: 0 (key), 1 (graph), 2 (index), 5 (flips).

    Returns:
    --------
    List[Dict[str, Any]]
        A list where each dictionary corresponds to a successfully processed entry
        from filtered_results. Each dictionary contains:
        {
            'hash_key': str,
            'graph_index': int,
            'graph': ig.Graph, # The graph object itself (optional based on need)
            'ordered_unique_edges': List[Tuple[int, int]], # Edges defining sign tuple order
            'signed_states_results': List[Tuple[Tuple[int, ...], CounterType[str]]], # List of (sign_tuple, state_vector)
            'pm_to_base_state': List[Tuple[List[Tuple[int, int]], str]], # Mapping from PM structure to base state string
            'applied_bit_flips': List[int] # Bit flips from Stage 2 filtering
        }
    """
    final_signed_analysis_results = [] # List to store the final results

    # --- Input data validation (Optional) ---
    if not isinstance(filtered_results, list):
        print("Error: Expected 'filtered_results' to be a list.")
        return [] # Return empty list
    if not filtered_results:
        print("Info: 'filtered_results' is empty. No graphs to process in Stage 3.")
        return [] # Return empty list

    print(f"\nStarting Stage 3: Performing 2^K analysis on {len(filtered_results)} filtered graph entries...")

    # --- Iterate through filtered results ---
    for entry_index, filter_entry in enumerate(filtered_results):
        # --- Input item structure check and data extraction ---
        try:
            # Check return tuple structure from check_quantum_states_with_bit_flips
            # Need at least indices 0, 1, 2, 5
            if not isinstance(filter_entry, tuple) or len(filter_entry) < 6:
                 print(f"  Skipping entry {entry_index}: Invalid format or insufficient elements (expected at least 6).")
                 continue

            hash_key = filter_entry[0]
            graph_obj = filter_entry[1]
            graph_idx = filter_entry[2]
            applied_flips_list = filter_entry[5] # Extract bit flip info (Index 5)

            # Validate applied_flips_list type (Optional)
            if not isinstance(applied_flips_list, list):
                 print(f"  Warning: Entry {entry_index} (Key='{hash_key}', Idx={graph_idx}): 'applied_flips_list' is not a list ({type(applied_flips_list)}). Storing as is.")

            # Validate graph_obj
            if not isinstance(graph_obj, ig.Graph):
                 print(f"  Skipping entry {entry_index} (Key='{hash_key}', Idx={graph_idx}): Invalid graph object type {type(graph_obj)}.")
                 continue

            print(f"  Processing filtered entry {entry_index}: Key='{hash_key}', Index={graph_idx}")

            # a. Get partitions
            U, V = get_bipartite_sets(graph_obj)
            # Check partition validity
            if not U or not V or len(U) != len(V):
                 print(f"    Skipping: Invalid partitions found (U:{len(U)}, V:{len(V)})")
                 continue

            # b. Find perfect matching structures (recalculate)
            pm_structures = find_all_perfect_matching_structures(graph_obj, U, V)
            # Check if PM structures were found
            if not pm_structures:
                 print(f"    Skipping: No perfect matching structures found for this graph.")
                 continue
            print(f"    Found {len(pm_structures)} PM structures.")

            # c. Calculate state vectors and PM-state mapping
            # Call calculate_all_signed_states and get all three return values
            ordered_edges, signed_states_results, pm_to_base_state = \
                calculate_all_signed_states(graph_obj, U, V, pm_structures)

            # d. Store final result (without redundant 'pm_structures')
            if signed_states_results is not None: # Check calculation result
                final_signed_analysis_results.append({
                    'hash_key': hash_key,
                    'graph_index': graph_idx,
                    'graph': graph_obj,                       # Store graph object if needed downstream
                    'ordered_unique_edges': ordered_edges,    # Edges defining sign tuple order
                    'signed_states_results': signed_states_results, # List of (sign_tuple, state_counter)
                    'pm_to_base_state': pm_to_base_state,     # Mapping from PM structure to base state string
                    'applied_bit_flips': applied_flips_list   # Bit flips from Stage 2
                })
                print(f"    Successfully generated {len(signed_states_results)} (sign_tuple, state_vector) pairs.")
            else:
                print(f"    Warning: State calculation failed or resulted in no states.")

        except Exception as e:
            print(f"    ERROR processing entry {entry_index} (Key='{hash_key}', Idx={graph_idx}): {e}")
            # traceback.print_exc() # Uncomment for detailed debugging

    print(f"\nStage 3 finished. Final analysis results generated for {len(final_signed_analysis_results)} entries.")
    return final_signed_analysis_results


def has_odd_negative_coeffs(state_counter: CounterType[str]) -> bool:
    """Checks if the Counter has an odd number of negative coefficients."""
    # Get coefficients from state_counter.values() and count how many are negative
    negative_count = sum(1 for coeff in state_counter.values() if coeff < 0)
    # Return True if count is odd, False if even
    return negative_count % 2 != 0

def filter_results_by_odd_negatives(
    final_analysis_output: List[Dict[str, Any]],
    # ===> Added new optional parameter <===
    required_num_states: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Filters the final analysis results based on specified conditions
    applied to the state vectors (Counters).

    Conditions checked:
    1. State vector must have an odd number of negative coefficients.
    2. (Optional) State vector must have exactly `required_num_states` entries
       if `required_num_states` is not None.

    Keeps graph entries only if they contain at least one state vector
    satisfying ALL specified conditions. For kept entries, the state vector
    list ('signed_states_results') is also filtered accordingly.

    Args:
        final_analysis_output: The list of result dictionaries.
        required_num_states: If not None, only state vectors with exactly
                             this number of non-zero entries (keys) are kept,
                             in addition to the odd negative coefficient check.

    Returns:
        A new list containing only the filtered result dictionaries.
    """
    filtered_output = []
    if not final_analysis_output:
        return []

    # Create filter condition description string
    filter_desc = ["odd number of negative coefficients"]
    if required_num_states is not None:
        # Validate required_num_states (optional)
        if not isinstance(required_num_states, int) or required_num_states < 0:
             print(f"Warning: Invalid required_num_states ({required_num_states}). Must be a non-negative integer. Disabling this filter.", file=sys.stderr)
             required_num_states = None # Disable filter if invalid
        else:
             filter_desc.append(f"exactly {required_num_states} states")

    print(f"\nStarting filtering based on: [{', '.join(filter_desc)}] on {len(final_analysis_output)} graph entries...")

    for graph_result_dict in final_analysis_output:
        if 'signed_states_results' not in graph_result_dict or \
           not isinstance(graph_result_dict['signed_states_results'], list):
            # Print warning message (same as before)
            print(f"Warning: Skipping entry for key {graph_result_dict.get('hash_key', 'N/A')}, index {graph_result_dict.get('graph_index', 'N/A')} due to missing/invalid 'signed_states_results'.")
            continue

        qualifying_signed_states = []
        for sign_tuple, state_counter in graph_result_dict['signed_states_results']:

            # ===> Condition 1: Check for odd number of negative coefficients <===
            passes_odd_check = has_odd_negative_coeffs(state_counter)

            # ===> Condition 2: Check state count (if required_num_states is specified) <===
            # len(state_counter) returns the number of keys with non-zero coefficients
            passes_num_check = (required_num_states is None) or \
                               (len(state_counter) == required_num_states)

            # ===> Check if both conditions are satisfied <===
            if passes_odd_check and passes_num_check:
                qualifying_signed_states.append((sign_tuple, state_counter))

        # If there's at least one state vector in this graph that satisfies all conditions
        if qualifying_signed_states:
            new_graph_result = graph_result_dict.copy()
            # Replace 'signed_states_results' with the filtered list
            new_graph_result['signed_states_results'] = qualifying_signed_states
            filtered_output.append(new_graph_result)
            print(f"  Keeping entry Key='{graph_result_dict.get('hash_key', 'N/A')}', Index={graph_result_dict.get('graph_index', 'N/A')}: Found {len(qualifying_signed_states)} states meeting all criteria.")
        # else: # If no state vectors satisfy all conditions, discard this graph result

    print(f"Filtering complete. Kept {len(filtered_output)} graph entries.")
    return filtered_output

########################################################