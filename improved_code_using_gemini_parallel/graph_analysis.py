# graph_analysis.py
"""
Functions for analyzing and processing generated EPM graphs.
Includes graph conversion, isomorphism checks, grouping, and SCC filtering.
"""

import igraph as ig
import numpy as np
import hashlib
from typing import List, Tuple, Dict, Any, Iterator, Set, Optional # For type hinting

# --- Graph Conversion ---

import igraph as ig
import numpy as np
from typing import List, Tuple, Any, Dict # 타입 힌트를 위해 추가

import igraph as ig
import numpy as np
from typing import List, Tuple, Any, Dict

def EPM_digraph_from_EPM_bipartite_graph_igraph(B: ig.Graph) -> ig.Graph:
    """
    Converts an EPM bipartite graph (B) from igraph format to a directed graph (D).
    Edges in D originate from system/ancilla nodes ('i') and target nodes ('j')
    determined by sculpting node indices ('k' where j=k). The edge direction
    is swapped from the original interpretation (i -> j instead of j -> i).

    Parameters:
    -----------
    B : igraph.Graph
        The EPM bipartite graph object to convert.
        Must contain 'category' node attribute ('system_nodes', 'ancilla_nodes',
        'sculpting_nodes') and 'weight' edge attribute.

    Returns:
    --------
    igraph.Graph
        The generated directed EPM graph object (D).
        Contains 'category' and 'name' node attributes, and 'weight' edge attribute.

    Raises:
    -------
    KeyError:
        If required attributes 'category' or 'weight' are missing in graph B.
    ValueError:
        If the graph structure is inconsistent or validation fails.
    Exception:
        For any other unexpected errors during conversion.
    """
    try: # Start error handling
        # --- 1. Classify nodes and count ---
        # Store the *index* of nodes belonging to each category ('system_nodes', 'ancilla_nodes', 'sculpting_nodes') in lists.
        system_nodes = [v.index for v in B.vs if v['category'] == "system_nodes"]
        ancilla_nodes = [v.index for v in B.vs if v['category'] == "ancilla_nodes"]
        sculpting_nodes = [v.index for v in B.vs if v['category'] == "sculpting_nodes"]

        # Calculate the number of nodes for each category.
        num_system = len(system_nodes)
        num_ancilla = len(ancilla_nodes)
        num_sculpting = len(sculpting_nodes)
        # Get the total number of nodes in the input graph B.
        num_total_bipartite = B.vcount()
        # The number of nodes in the final directed graph D is the sum of system and ancilla nodes
        num_total_digraph = num_system + num_ancilla

        # --- 2. Validations ---
        # Check if the sum of classified node counts matches the total node count in B.
        if num_total_bipartite != num_system + num_ancilla + num_sculpting:
            raise ValueError("Sum of node counts does not match the total number of nodes in the bipartite graph.")
        # Warn if the sculpting node count differs from the final graph D's node count (may relate to j=k mapping)
        if num_sculpting != num_total_digraph:
            print(f"Warning: Sculpting node count ({num_sculpting}) does not match system+ancilla count ({num_total_digraph}). Ensure the mapping 'j=k' and its implications are intended.")

        # --- 3. Calculate and reorder adjacency matrix ---
        # Node order: system -> ancilla -> sculpting
        ordered_vertices = system_nodes + ancilla_nodes + sculpting_nodes
        # Calculate weighted adjacency matrix (convert to NumPy array)
        adj_matrix_B = np.array(B.get_adjacency(attribute="weight").data)
        # Reorder the adjacency matrix according to ordered_vertices
        reordered_adj_matrix = adj_matrix_B[np.ix_(ordered_vertices, ordered_vertices)]

        # --- 4. Extract edge information for directed graph D ---
        # Extract the submatrix used for calculating edges in graph D
        # Rows: system+ancilla nodes, Columns: sculpting nodes
        adj_matrix_D_sub = reordered_adj_matrix[:num_total_digraph, num_total_digraph:]

        # --- 5. Initialize directed graph D and set node attributes ---
        # Create D as a directed graph with num_total_digraph nodes
        D = ig.Graph(n=num_total_digraph, directed=True)
        # Set 'category' and 'name' attributes for the nodes in graph D
        categories = ["system_nodes"] * num_system + ["ancilla_nodes"] * num_ancilla
        node_names = [f"S_{i}" for i in range(num_system)] + [f"A_{i}" for i in range(num_ancilla)]
        D.vs["category"] = categories
        D.vs["name"] = node_names

        # --- 6. Add Directed Edges (Simple Source/Target Swap) ---
        edges: List[Tuple[int, int]] = []
        weights: List[float] = []

        # Loops and variable setup same as original logic
        for i in range(num_total_digraph): # Originally: Target role -> Now: Source role (system/ancilla node index)
            for k in range(num_sculpting): # Sculpting node index
                # Map sculpting index k to node j in D (Originally: Source -> Now: Target)
                j = k
                if j < num_total_digraph: # Ensure target node index j is valid for D
                    # Weight is from connection between node i (now source) and sculpting node k
                    weight = adj_matrix_D_sub[i, k]
                    if weight != 0:
                        # *** Change Point: Edge direction changed to (i, j) ***
                        # i.e., add edge from system/ancilla node i to node j corresponding to sculpting node k
                        edges.append((i, j)) # Edge from i to j
                        weights.append(weight)
                else:
                     # Case where j is out of bounds (can happen if num_sculpting > num_total_digraph and k >= num_total_digraph)
                     # In this case, there is no valid target j reachable from source i, so the edge is not created.
                     print(f"Warning: Sculpting node index {k} maps to digraph index {j}, which is out of bounds ({num_total_digraph}). Skipping potential edge from source {i} targetted at {j}.")

        # --- 7. Final application of edges and weights to graph D ---
        if edges:
            D.add_edges(edges)
            D.es["weight"] = weights

        # --- 8. Return result ---
        return D

    # --- Error Handling ---
    except KeyError as e:
        print(f"Error: Missing required attribute in input graph B: {e}")
        raise
    except ValueError as e:
        print(f"Error: Graph structure validation failed: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during digraph conversion: {e}")
        raise

# def EPM_digraph_from_EPM_bipartite_graph_igraph(B: ig.Graph) -> ig.Graph:
#     """
#     Converts an EPM bipartite graph (B) from igraph format to a directed graph (D).
    
#     Parameters:
#     -----------
#     B : igraph.Graph
#         The EPM bipartite graph in igraph format.
#         Required node attributes: 'category' ('system_nodes', 'ancilla_nodes', 'sculpting_nodes')
#         Required edge attributes: 'weight'
        
#     Returns:
#     --------
#     igraph.Graph
#         A directed graph where nodes are system and ancilla nodes from B,
#         and edges represent relationships derived from connections to sculpting nodes.
        
#     Raises:
#     -------
#     KeyError: If required node/edge attributes are missing
#     ValueError: If graph structure validation fails
#     Exception: For other unexpected errors
#     """
#     try: # Start error handling
#         # --- 1. Classify nodes and count them ---
#         # Iterate through all vertices in graph B and check their 'category' attribute
#         # Store the indices of nodes in each category ('system_nodes', 'ancilla_nodes', 'sculpting_nodes')
#         system_nodes = [v.index for v in B.vs if v['category'] == "system_nodes"]
#         ancilla_nodes = [v.index for v in B.vs if v['category'] == "ancilla_nodes"]
#         sculpting_nodes = [v.index for v in B.vs if v['category'] == "sculpting_nodes"]

#         # Calculate the number of nodes in each category
#         num_system = len(system_nodes)
#         num_ancilla = len(ancilla_nodes)
#         num_sculpting = len(sculpting_nodes)
#         # Get the total number of nodes in the input bipartite graph B
#         num_total_bipartite = B.vcount()
#         # Calculate the number of nodes for the final directed graph D
#         # Assuming only system and ancilla nodes will become nodes in D
#         num_total_digraph = num_system + num_ancilla

#         # --- 2. Validation checks ---
#         # Verify that the sum of classified nodes matches the total number of nodes in B
#         if num_total_bipartite != num_system + num_ancilla + num_sculpting:
#             raise ValueError("Sum of node counts does not match the total number of nodes in the bipartite graph.")

#         # Check if the number of sculpting nodes equals the total number of nodes in the final graph D
#         # This is a specific assumption of the EPM model; output a warning if not satisfied
#         if num_sculpting != num_total_digraph:
#             print(f"Warning: Sculpting node count ({num_sculpting}) does not match system+ancilla count ({num_total_digraph}). Ensure this is intended.")
#             # Could raise ValueError here for strict validation if needed

#         # --- 3. Calculate and reorder adjacency matrix ---
#         # Create a list of node indices in a specific order (system -> ancilla -> sculpting)
#         # This order will be used to reorder the adjacency matrix
#         ordered_vertices = system_nodes + ancilla_nodes + sculpting_nodes

#         # Calculate the weighted adjacency matrix of input graph B
#         # B.get_adjacency(attribute="weight") returns the adjacency matrix with weights (as igraph.Matrix)
#         # .data accesses the actual data, and np.array() converts it to a NumPy array for easier manipulation
#         # At this point, the rows/columns of adj_matrix_B follow B's original node index order (0 to num_total_bipartite-1)
#         adj_matrix_B = np.array(B.get_adjacency(attribute="weight").data)

#         # Use NumPy's indexing feature (np.ix_) to reorder the adjacency matrix rows and columns according to ordered_vertices
#         # reordered_adj_matrix now has rows/columns ordered as system -> ancilla -> sculpting
#         reordered_adj_matrix = adj_matrix_B[np.ix_(ordered_vertices, ordered_vertices)]

#         # --- 4. Extract edge information for directed graph D ---
#         # Extract the submatrix from the reordered adjacency matrix needed to determine edges in graph D
#         # Rows: portion corresponding to system and ancilla nodes (first num_total_digraph rows)
#         # Columns: portion corresponding to sculpting nodes (from index num_total_digraph to the end)
#         # adj_matrix_D_sub[i, k] represents the edge weight between the i-th system/ancilla node and k-th sculpting node in the reordered matrix
#         adj_matrix_D_sub = reordered_adj_matrix[:num_total_digraph, num_total_digraph:]

#         # --- 5. Initialize directed graph D and set node attributes ---
#         # Initialize the final directed graph D with num_total_digraph nodes and directed=True
#         D = ig.Graph(n=num_total_digraph, directed=True)

#         # Assign 'category' and 'name' attributes to each node in D (indices 0 to num_total_digraph-1)
#         # Node names are created in the format 'S_i' (system) and 'A_i' (ancilla) for distinction
#         categories = ["system_nodes"] * num_system + ["ancilla_nodes"] * num_ancilla
#         node_names = [f"S_{i}" for i in range(num_system)] + [f"A_{i}" for i in range(num_ancilla)]
#         D.vs["category"] = categories
#         D.vs["name"] = node_names

#         # --- 6. Add directed edges ---
#         # Initialize lists to store edge information (source, target) tuples and weights
#         edges: List[Tuple[int, int]] = []
#         weights: List[float] = []

#         # Edge direction in D is from 'j' to 'i'
#         # Important assumption: the k-th sculpting node corresponds to the j-th node (source) in D (j = k)
#         # This mapping may need to be changed depending on the EPM model definition being used

#         # Outer loop: iterate through each node i (0 to num_total_digraph-1) in D. This node becomes the edge *target*
#         for i in range(num_total_digraph):
#             # Inner loop: iterate through each column k (0 to num_sculpting-1) in adj_matrix_D_sub, corresponding to sculpting nodes
#             for k in range(num_sculpting):
#                 # Map sculpting node index k to potential *source* node index j in D (currently direct mapping: j = k)
#                 j = k
#                 # Check if the mapped source node j is within the valid range of node indices in D
#                 # (j could be out of bounds if num_sculpting > num_total_digraph)
#                 if j < num_total_digraph:
#                     # Get the weight at position (i, k) in the submatrix
#                     # This value represents the connection weight between the i-th sys/anc node and k-th sculpting node in the original B graph
#                     weight = adj_matrix_D_sub[i, k]
#                     # Only add an edge if the weight is non-zero (i.e., there was a connection in B)
#                     if weight != 0:
#                         # Set edge direction from j to i: (source=j, target=i)
#                         edges.append((j, i))
#                         # Store the weight for this edge
#                         weights.append(weight)
#                 else:
#                      # If j is out of bounds (can happen when num_sculpting > num_total_digraph and k >= num_total_digraph), 
#                      # output a warning and skip this potential edge
#                      print(f"Warning: Sculpting node index {k} maps to digraph index {j}, which is out of bounds ({num_total_digraph}). Skipping potential edge.")

#         # --- 7. Apply edges and weights to D graph ---
#         # If there's at least one edge to add (edges list is not empty)
#         if edges:
#             # Add all collected edges to D graph at once
#             D.add_edges(edges)
#             # Assign the collected weights as the 'weight' attribute to the added edges
#             D.es["weight"] = weights

#         # --- 8. Return result ---
#         # Return the completed directed graph D
#         return D

#     # --- Error handling ---
#     except KeyError as e: # Occurs when 'category' or 'weight' attributes are missing
#         print(f"Error: Missing required attribute in input graph B: {e}")
#         raise # Re-raise the error so the caller is aware
#     except ValueError as e: # Occurs when validation checks fail
#         print(f"Error: Graph structure validation failed: {e}")
#         raise
#     except Exception as e: # For any other unexpected errors
#         print(f"An unexpected error occurred during digraph conversion: {e}")
#         raise

# # --- Isomorphism and Grouping ---

# # def canonical_form_without_weights(ig_graph: ig.Graph) -> tuple:
# #     """
# #     Generates a canonical representation (adjacency matrix) of the graph structure,
# #     ignoring edge weights and node/edge attributes other than connectivity.

# #     Uses igraph's canonical permutation.

# #     Parameters:
# #     -----------
# #     ig_graph : igraph.Graph
# #         The input graph.

# #     Returns:
# #     --------
# #     tuple
# #         A tuple representation of the permuted adjacency matrix, suitable for hashing or comparison.
# #     """
# #     if not isinstance(ig_graph, ig.Graph):
# #         raise TypeError("Input must be an igraph.Graph object.")
# #     if ig_graph.vcount() == 0:
# #         return tuple() # Handle empty graph

# #     try:
# #         # Get the canonical permutation based on graph structure only
# #         # color/label args are omitted to ignore attributes
# #         perm: List[int] = ig_graph.canonical_permutation()

# #         # Apply the permutation to the vertices
# #         permuted_graph: ig.Graph = ig_graph.permute_vertices(perm)

# #         # Get the adjacency matrix of the permuted graph
# #         # Use bool type to explicitly ignore weights
# #         adj_matrix: List[List[bool]] = permuted_graph.get_adjacency(type=ig.ADJ_BOOL).data

# #         # Convert the adjacency matrix (list of lists) to a tuple of tuples for immutability
# #         return tuple(map(tuple, adj_matrix))
# #     except Exception as e:
# #         print(f"Error generating canonical form for graph: {e}")
# #         raise

def canonical_form_without_weights(ig_graph: ig.Graph) -> tuple:
    """
    Generates a canonical representation (adjacency matrix) of the graph structure,
    ignoring edge weights and node/edge attributes other than connectivity.

    Uses igraph's canonical permutation. (Corrected version)

    Parameters:
    -----------
    ig_graph : igraph.Graph
        The input graph.

    Returns:
    --------
    tuple
        A tuple representation of the permuted boolean adjacency matrix,
        suitable for hashing or comparison.
    """
    if not isinstance(ig_graph, ig.Graph):
        raise TypeError("Input must be an igraph.Graph object.")
    if ig_graph.vcount() == 0:
        return tuple() # Handle empty graph

    try:
        # Get the canonical permutation based on graph structure only
        perm: List[int] = ig_graph.canonical_permutation()

        # Apply the permutation to the vertices
        permuted_graph: ig.Graph = ig_graph.permute_vertices(perm)

        # Get the default numerical adjacency matrix
        adj_matrix_obj = permuted_graph.get_adjacency()
        adj_matrix_data: List[List[float]] = adj_matrix_obj.data # Usually float

        # Convert the numerical data to boolean (True if edge exists, False otherwise)
        # An edge exists if the corresponding value is non-zero.
        adj_matrix_bool_data: List[List[bool]] = [
            [bool(value != 0) for value in row] for row in adj_matrix_data
        ]

        # Convert the boolean adjacency matrix (list of lists) to a tuple of tuples for immutability
        return tuple(map(tuple, adj_matrix_bool_data))

    except AttributeError as ae:
        # Catch potential issues with get_adjacency() or data attribute if API changed drastically
        print(f"Error accessing adjacency data or permutation: {ae}")
        raise
    except Exception as e:
        print(f"Error generating canonical form for graph: {e}")
        raise


# --- Helper function used by process_and_group_by_canonical_form ---
def generate_hash_from_canonical_form(canonical_form: tuple) -> str:
    """
    Generates a SHA-256 hash from the canonical form tuple.

    Parameters:
    -----------
    canonical_form : tuple
        The tuple representation of the canonical adjacency matrix.

    Returns:
    --------
    str
        The hexadecimal SHA-256 hash string.
    """
    # Convert the tuple of tuples to a string representation
    canonical_str = str(canonical_form)
    # Encode the string to bytes and generate the hash
    hash_object = hashlib.sha256(canonical_str.encode('utf-8'))
    return hash_object.hexdigest()


def process_and_group_by_canonical_form(graph_iterator: Iterator[ig.Graph]) -> Dict[str, List[ig.Graph]]:
    """
    Processes an iterator of graphs, groups them based on their weight-agnostic
    canonical form (isomorphism), using hashes as keys.

    Parameters:
    -----------
    graph_iterator : Iterator[igraph.Graph]
        An iterator yielding igraph.Graph objects (e.g., from a generator).

    Returns:
    --------
    Dict[str, List[igraph.Graph]]
        A dictionary where keys are the hash strings of the canonical forms,
        and values are lists of graphs belonging to that isomorphism class (ignoring weights).
    """
    canonical_groups: Dict[str, List[ig.Graph]] = {}
    processed_count = 0
    error_count = 0

    for graph in graph_iterator:
        processed_count += 1
        try:
            # Generate canonical form (ignoring weights)
            c_form = canonical_form_without_weights(graph)
            # Generate hash from the canonical form
            c_hash = generate_hash_from_canonical_form(c_form)

            # Add the graph to the corresponding group
            if c_hash not in canonical_groups:
                canonical_groups[c_hash] = []
            canonical_groups[c_hash].append(graph)

        except Exception as e:
            error_count += 1
            print(f"Error processing graph number {processed_count}: {e}. Skipping this graph.")
            # Optionally log more details about the graph that caused the error
            # print(f"Graph details (first few nodes/edges): {graph.summary()}")

    print(f"Grouping complete. Processed {processed_count} graphs, encountered {error_count} errors.")
    print(f"Found {len(canonical_groups)} unique structural isomorphism classes (ignoring weights).")
    return canonical_groups


def extract_unique_bigraphs_with_weights_igraph(graph_list: List[ig.Graph]) -> List[ig.Graph]:
    """
    Extracts unique bipartite graphs from a list, considering edge weights
    for isomorphism checking using igraph's VF2 algorithm.

    Assumes all graphs in the list are structurally isomorphic (weight-agnostic).

    Parameters:
    -----------
    graph_list : List[igraph.Graph]
        A list of bipartite graphs, typically belonging to the same
        weight-agnostic isomorphism class.

    Returns:
    --------
    List[igraph.Graph]
        A list containing only the unique bipartite graphs from the input list,
        where uniqueness is determined by weighted isomorphism.
    """
    if not graph_list:
        return []

    unique_graphs_representatives: List[ig.Graph] = []
    processed_count = 0
    unique_count = 0

    for new_graph in graph_list:
        processed_count += 1
        is_unique = True

        # Check basic properties first for quick filtering (optional but can speed up)
        # Since they are assumed to be in the same structural group, vcount/ecount should match.
        # if new_graph.vcount() != graph_list[0].vcount() or new_graph.ecount() != graph_list[0].ecount():
        #     print("Warning: Graph list contains graphs with different sizes. This function assumes structural isomorphism.")
        #     continue # Or handle error as appropriate

        # Extract edge weights (use default 1 if 'weight' attribute is missing)
        new_weights = new_graph.es["weight"] if "weight" in new_graph.edge_attributes() else [1.0] * new_graph.ecount()

        # Compare against representatives of already found unique weighted graphs
        for existing_representative in unique_graphs_representatives:
            existing_weights = existing_representative.es["weight"] if "weight" in existing_representative.edge_attributes() else [1.0] * existing_representative.ecount()

            # Check weighted isomorphism using VF2 algorithm
            # edge_color1/2 arguments compare edge attributes (weights in this case)
            try:
                # Ensure weights are comparable (e.g., numbers)
                if new_graph.isomorphic_vf2(existing_representative,
                                           edge_color1=new_weights,
                                           edge_color2=existing_weights):
                    # Found an isomorphic graph, so new_graph is not unique
                    is_unique = False
                    break # No need to compare with other representatives
            except TypeError as e:
                 print(f"Warning: Could not compare weights during isomorphism check: {e}. Assuming graphs are different.")
                 # Treat as non-isomorphic if weights are not comparable, or handle differently
            except Exception as e:
                 print(f"Error during VF2 isomorphism check: {e}. Assuming graphs are different.")


        # If no isomorphic representative was found, add this graph as a new unique representative
        if is_unique:
            unique_graphs_representatives.append(new_graph)
            unique_count += 1

    # print(f"Processed {processed_count} graphs. Found {unique_count} unique weighted isomorphism classes.")
    return unique_graphs_representatives


def extract_unique_bigraphs_from_groups_igraph(grouped_graphs: Dict[str, List[ig.Graph]]) -> Dict[str, List[ig.Graph]]:
    """
    Processes a dictionary of graph groups (grouped by weight-agnostic canonical form)
    and extracts the unique graphs within each group based on weighted isomorphism.

    Parameters:
    -----------
    grouped_graphs : Dict[str, List[igraph.Graph]]
        A dictionary where keys are hashes of weight-agnostic canonical forms,
        and values are lists of graphs belonging to that class.

    Returns:
    --------
    Dict[str, List[igraph.Graph]]
        A dictionary with the same keys, but where the values are lists containing
        only the unique representatives (based on weighted isomorphism) from the original lists.
    """
    unique_results: Dict[str, List[ig.Graph]] = {}
    total_unique_weighted_graphs = 0

    print(f"Extracting unique weighted graphs from {len(grouped_graphs)} structural groups...")
    for key, graph_list in grouped_graphs.items():
        # Ensure all items in the list are valid igraph Graph objects
        valid_graphs = [g for g in graph_list if isinstance(g, ig.Graph)]
        if len(valid_graphs) != len(graph_list):
            print(f"Warning: Group '{key}' contained {len(graph_list) - len(valid_graphs)} non-Graph items.")

        if valid_graphs:
            # Extract unique graphs considering weights within this group
            unique_representatives = extract_unique_bigraphs_with_weights_igraph(valid_graphs)
            unique_results[key] = unique_representatives
            total_unique_weighted_graphs += len(unique_representatives)
            # print(f"Group '{key}': Input {len(valid_graphs)} graphs -> Found {len(unique_representatives)} unique weighted graphs.")
        else:
            unique_results[key] = [] # Keep the key but with an empty list if no valid graphs
            # print(f"Group '{key}': Contained no valid graphs.")

    print(f"Extraction complete. Total unique weighted graphs found across all groups: {total_unique_weighted_graphs}")
    return unique_results


# --- SCC Filtering ---

def is_single_scc_igraph(graph: ig.Graph) -> bool:
    """
    Checks if a directed graph consists of a single Strongly Connected Component (SCC).

    Parameters:
    -----------
    graph : igraph.Graph
        The directed graph to check. Must be a directed graph.

    Returns:
    --------
    bool
        True if the graph has exactly one SCC and it contains all vertices, False otherwise.

    Raises:
    ------
    TypeError:
        If the input graph is not directed.
    """
    if not graph.is_directed():
        raise TypeError("Input graph must be directed to check for SCCs.")
    if graph.vcount() == 0:
        return True # An empty graph can be considered a single SCC (or handle as needed)

    try:
        # Find strongly connected components. Returns a VertexClustering object.
        sccs = graph.connected_components(mode="strong")

        # Check if there is exactly one component and it includes all vertices
        return len(sccs) == 1 # and len(sccs[0]) == graph.vcount() # This second check is implicit in len(sccs) == 1 for non-empty graphs

    except Exception as e:
        print(f"Error checking SCC for graph: {e}")
        # Decide how to handle errors: re-raise, return False, etc.
        return False # Assume not a single SCC if an error occurs


def filter_groups_by_scc_igraph(grouped_graphs: Dict[str, List[ig.Graph]]) -> Dict[str, List[ig.Graph]]:
    """
    Filters the grouped graphs, keeping only those groups where the corresponding
    directed graph (derived from the first graph in the group) forms a single SCC.

    Parameters:
    -----------
    grouped_graphs : Dict[str, List[igraph.Graph]]
        Dictionary of graph groups (key: hash, value: list of bipartite igraph.Graph).

    Returns:
    --------
    Dict[str, List[igraph.Graph]]
        A filtered dictionary containing only the groups that satisfy the single SCC condition.
    """
    filtered_groups: Dict[str, List[ig.Graph]] = {}
    groups_checked = 0
    groups_passed = 0
    error_count = 0

    print(f"Filtering {len(grouped_graphs)} graph groups based on SCC condition...")
    for key, graph_list in grouped_graphs.items():
        groups_checked += 1
        if not graph_list: # Skip empty groups
            # print(f"Group '{key}' is empty. Skipping.")
            continue

        try:
            # Use the first graph in the list as representative for the SCC check
            first_graph = graph_list[0]
            if not isinstance(first_graph, ig.Graph):
                 print(f"Warning: First item in group '{key}' is not an igraph.Graph. Skipping group.")
                 error_count += 1
                 continue

            # 1. Convert the representative bipartite graph to its directed counterpart
            D = EPM_digraph_from_EPM_bipartite_graph_igraph(first_graph)

            # 2. Check if the directed graph D is a single SCC
            if is_single_scc_igraph(D):
                # If it satisfies the condition, keep the entire original group
                filtered_groups[key] = graph_list
                groups_passed += 1
            # else: # Optional: Log groups that failed the SCC check
            #     print(f"Group '{key}' failed SCC check.")

        except Exception as e:
            error_count += 1
            print(f"Error processing group '{key}' for SCC check: {e}. Skipping group.")
            # Optionally log graph details: print(f"Graph summary: {graph_list[0].summary()}")

    print(f"SCC Filtering complete. Checked {groups_checked} groups. Passed: {groups_passed}. Errors: {error_count}.")
    return filtered_groups

