# utils.py
"""
Common utility functions used across different modules.
"""

import igraph as ig
import networkx as nx
from typing import Dict, Any, Set, List, Tuple # For type hinting

def igraph_to_networkx(g_igraph: ig.Graph) -> nx.Graph | nx.DiGraph: # 반환 타입 변경
    """
    Converts an igraph Graph object to a NetworkX Graph or DiGraph object,
    preserving directedness.

    Parameters:
    -----------
    g_igraph : igraph.Graph
        The igraph graph to convert. Node and edge attributes are preserved.
        Assumes nodes have a 'name' attribute for unique identification in NetworkX.

    Returns:
    --------
    networkx.Graph or networkx.DiGraph
        The converted NetworkX graph, matching the directedness of the input.
    """
    # Check if the igraph graph is directed and choose the appropriate NetworkX class
    if g_igraph.is_directed():
        G_nx = nx.DiGraph() # Use DiGraph for directed graphs
    else:
        G_nx = nx.Graph()   # Use Graph for undirected graphs

    # Add nodes and their attributes
    if not g_igraph.vs: # Handle empty graph
        return G_nx

    # Ensure 'name' attribute exists or handle node identification differently
    # Option 1: Assume 'name' exists (as in original code)
    try:
        for v in g_igraph.vs:
            node_name = v["name"]
            node_attrs: Dict[str, Any] = {attr: v[attr] for attr in v.attribute_names() if attr != 'name'}
            G_nx.add_node(node_name, **node_attrs)
    except KeyError:
         # Option 2: Use igraph index as NetworkX node ID if 'name' is missing
         print("Warning: Node 'name' attribute missing. Using igraph vertex index as node ID.")
         G_nx = nx.DiGraph() if g_igraph.is_directed() else nx.Graph() # Re-initialize in case loop failed midway
         for v in g_igraph.vs:
             node_id = v.index # Use index as node identifier
             node_attrs: Dict[str, Any] = {attr: v[attr] for attr in v.attribute_names()}
             G_nx.add_node(node_id, **node_attrs)


    # Add edges and their attributes
    if not g_igraph.es: # Handle graph with no edges
         return G_nx

    # Adjust edge addition based on how nodes were added (name vs index)
    try: # Assuming nodes were added using 'name'
        source_node_map = {v.index: v["name"] for v in g_igraph.vs}
        target_node_map = source_node_map
    except KeyError: # Assuming nodes were added using index
         source_node_map = {v.index: v.index for v in g_igraph.vs}
         target_node_map = source_node_map

    for e in g_igraph.es:
        try:
            source_node = source_node_map[e.source]
            target_node = target_node_map[e.target]
            edge_attrs: Dict[str, Any] = {attr: e[attr] for attr in e.attribute_names()}
            G_nx.add_edge(source_node, target_node, **edge_attrs)
        except KeyError as ke:
             print(f"Warning: Could not find node for edge source {e.source} or target {e.target} in map. Skipping edge. Error: {ke}")
        except Exception as ex:
             print(f"Error adding edge ({e.source}, {e.target}): {ex}")


    return G_nx

def networkx_to_igraph(G_nx: nx.Graph) -> ig.Graph:
    """
    Converts a NetworkX Graph object to an igraph Graph object.

    Parameters:
    -----------
    G_nx : networkx.Graph
        The NetworkX graph to convert. Node and edge attributes are preserved.

    Returns:
    --------
    igraph.Graph
        The converted igraph graph. Node 'name' attribute is set from NetworkX node identifiers.
    """
    g_igraph = ig.Graph(directed=G_nx.is_directed()) # Create directed/undirected based on input

    # Add vertices and 'name' attribute
    node_names: List[Any] = list(G_nx.nodes())
    g_igraph.add_vertices(len(node_names))
    g_igraph.vs["name"] = node_names # Use node identifier as 'name'

    # Create a mapping from node name to igraph vertex index for edge creation
    node_name_to_index: Dict[Any, int] = {name: idx for idx, name in enumerate(node_names)}

    # Add node attributes (excluding 'name' which is already set)
    # Get all unique attribute keys across all nodes
    node_attr_keys: Set[str] = set().union(*(d.keys() for _, d in G_nx.nodes(data=True)))
    for attr in node_attr_keys:
        # Extract attribute values for all nodes, handling missing attributes
        # Using G_nx.nodes[n].get(attr) returns None if attribute is missing
        attr_values = [G_nx.nodes[n].get(attr) for n in node_names]
        try:
             g_igraph.vs[attr] = attr_values
        except TypeError as e:
             print(f"Warning: Could not assign node attribute '{attr}' due to mixed types or unsupported type: {e}. Attribute skipped.")


    # Add edges
    edges: List[Tuple[int, int]] = [
        (node_name_to_index[u], node_name_to_index[v]) for u, v in G_nx.edges()
    ]
    g_igraph.add_edges(edges)

    # Add edge attributes
    # Get all unique attribute keys across all edges
    edge_attr_keys: Set[str] = set().union(*(d.keys() for _, _, d in G_nx.edges(data=True)))
    for attr in edge_attr_keys:
         # Extract attribute values for all edges in the order they were added
         attr_values = [G_nx.get_edge_data(u, v).get(attr) for u, v in G_nx.edges()]
         try:
              g_igraph.es[attr] = attr_values
         except TypeError as e:
              print(f"Warning: Could not assign edge attribute '{attr}' due to mixed types or unsupported type: {e}. Attribute skipped.")


    return g_igraph

# Add any other general utility functions here if needed
