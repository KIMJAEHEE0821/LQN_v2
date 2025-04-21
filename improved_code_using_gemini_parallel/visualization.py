# visualization.py
"""
Functions for visualizing EPM bipartite and directed graphs using
matplotlib and networkx.
"""
import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig # Needed for type hinting
from typing import Optional, Dict, List, Any, Tuple
from math import isclose

# Import the conversion utility from utils.py (Ensure this path is correct)
try:
    from utils import igraph_to_networkx
except ImportError:
    print("Error: Could not import 'igraph_to_networkx' from 'utils'.")
    print("Ensure utils.py is in the Python path.")
    # Define a dummy function to avoid crashing if utils is missing,
    # but visualization will likely fail.
    def igraph_to_networkx(g: ig.Graph) -> nx.Graph:
        print("Warning: Using dummy igraph_to_networkx function.")
        G_nx = nx.Graph() # Return an empty graph
        if isinstance(g, ig.Graph): # Basic check
            for v in g.vs: G_nx.add_node(v.index, **v.attributes())
            for e in g.es: G_nx.add_edge(e.source, e.target, **e.attributes())
        return G_nx


def Draw_EPM_bipartite_graph(
    B_igraph: ig.Graph,
    title: str = "EPM Bipartite Graph",
    ax: Optional[plt.Axes] = None # <<< ADDED: Optional ax parameter
    ):
    """
    Draws the EPM bipartite graph using NetworkX and Matplotlib.
    MODIFIED to accept an optional Axes object for external plotting control.

    Nodes are positioned based on category and colored. Edges are colored by weight.

    Parameters:
    -----------
    B_igraph : igraph.Graph
        The EPM bipartite graph (igraph object). Expected attributes:
        'category', 'name' (for labels), 'weight' (for edges).
    title : str, optional
        The title for the plot.
    ax : matplotlib.axes.Axes, optional
        The Axes object to draw on. If None, a new figure and axes are created.

    Raises:
    ------
    TypeError: If input is not an igraph.Graph.
    KeyError: If required node attribute 'category' is missing.
    Exception: For other plotting related errors.
    """
    if not isinstance(B_igraph, ig.Graph):
        raise TypeError("Input must be an igraph.Graph object.")

    # print(f"Attempting to draw bipartite graph: {B_igraph.summary()}") # Uncomment for debugging

    # --- Figure and Axes Handling ---
    if ax is None:
        # If no axes provided, create figure and axes internally
        # Use dynamic figsize calculation from the original code example
        num_nodes_for_size = B_igraph.vcount()
        dynamic_height = max(6, num_nodes_for_size * 0.5)
        fig, ax = plt.subplots(figsize=(8, dynamic_height)) # Use example figsize
    else:
        # Use the provided axes object
        fig = ax.figure # Get figure from provided axes

    try:
        # --- Graph Conversion ---
        B_nx = igraph_to_networkx(B_igraph) # Assumes this returns nx.Graph for bipartite

        # --- Prepare node positions and colors (Original Logic) ---
        pos: Dict[Any, Tuple[float, float]] = {}
        node_colors: List[str] = []
        node_labels: Dict[Any, str] = {} # Prepare labels for nx.draw
        system_nodes: List[Any] = []
        ancilla_nodes: List[Any] = []
        sculpting_nodes: List[Any] = []

        # Check for necessary 'category' attribute
        if not B_nx.nodes:
             print("Graph is empty, nothing to draw.")
             ax.set_axis_off()
             return
        if 'category' not in B_nx.nodes[list(B_nx.nodes())[0]]:
             raise KeyError("Node attribute 'category' is required for positioning and coloring.")

        for node, data in B_nx.nodes(data=True):
            category = data.get('category', 'unknown')
            # Use 'name' attribute if available, else node id/index
            node_labels[node] = str(data.get('name', node))
            if category == 'system_nodes':
                system_nodes.append(node)
                node_colors.append('lightblue')
            elif category == 'ancilla_nodes':
                ancilla_nodes.append(node)
                node_colors.append('lightcoral')
            elif category == 'sculpting_nodes':
                sculpting_nodes.append(node)
                node_colors.append('lightgreen')
            else:
                print(f"Warning: Node '{node}' has unknown category '{category}'. Assigning default.")
                sculpting_nodes.append(node) # Group with sculpting for position
                node_colors.append('grey')

        # --- Assign positions (Original Logic: Manual Layout) ---
        # Sort nodes for consistent vertical ordering
        try:
            system_nodes.sort()
            ancilla_nodes.sort()
            sculpting_nodes.sort()
        except TypeError:
            print("Warning: Could not sort nodes for positioning. Layout may be inconsistent.")

        y_pos_sys = {node: -index for index, node in enumerate(system_nodes)}
        y_pos_anc = {node: -(index + len(system_nodes)) for index, node in enumerate(ancilla_nodes)}
        # Simple vertical stacking for sculpting nodes on the right
        y_pos_sculpt = {node: -index for index, node in enumerate(sculpting_nodes)}

        pos.update((node, (0, y_pos_sys.get(node, 0))) for node in system_nodes)
        pos.update((node, (0, y_pos_anc.get(node, 0))) for node in ancilla_nodes)
        pos.update((node, (1, y_pos_sculpt.get(node, 0))) for node in sculpting_nodes) # x=1 for right side


        # --- Prepare edge colors (Original Logic) ---
        edge_colors: List[str] = []
        if B_nx.number_of_edges() > 0:
             first_edge_tuple = list(B_nx.edges())[0]
             first_edge_data = B_nx.get_edge_data(*first_edge_tuple)
             if first_edge_data is None or 'weight' not in first_edge_data:
                  print("Warning: Edge attribute 'weight' not found. Using default color 'grey'.")
                  edge_colors = ['grey'] * B_nx.number_of_edges()
             else:
                  for u, v, data in B_nx.edges(data=True):
                      weight = data.get('weight', 0)
                      # Use isclose for float comparison
                      if isclose(weight, 1.0): edge_colors.append('red')
                      elif isclose(weight, 2.0): edge_colors.append('blue')
                      elif isclose(weight, 3.0): edge_colors.append('black')
                      else: edge_colors.append('grey')
        # else: edge_colors remains empty list []

        # --- Draw the graph using the provided or created ax ---
        nx.draw(
            B_nx,
            pos,
            ax=ax,                    # <<< Draw on the specified axes
            labels=node_labels,       # Pass prepared labels
            with_labels=True,
            node_color=node_colors,
            edge_color=edge_colors,
            node_size=700,            # Original node size
            font_size=10,             # Original font size
            width=1.5                 # Original edge width
        )
        ax.set_title(title)           # <<< Set title on the axes
        ax.set_axis_off()             # <<< Turn off axis on the axes
        # --- Remove plt.show() ---

    except KeyError as e:
        print(f"Error drawing bipartite graph: Missing required attribute {e}")
        if ax: ax.set_axis_off()
        # Depending on calling loop, might want to re-raise 'raise e'
    except Exception as e:
        print(f"An unexpected error occurred during bipartite graph drawing: {e}")
        if ax: ax.set_axis_off()
        # raise e


try:
    from utils import igraph_to_networkx
except ImportError:
    print("Warning: Could not import 'igraph_to_networkx' from 'utils'. Using dummy.")
    def igraph_to_networkx(g: ig.Graph) -> nx.DiGraph: # Assume DiGraph for directed
        print("Warning: Using dummy igraph_to_networkx function.")
        # Return an empty DiGraph to avoid downstream errors if possible
        G_nx = nx.DiGraph()
        if isinstance(g, ig.Graph): # Basic check if g is provided
            for v in g.vs:
                G_nx.add_node(v.index, **v.attributes())
            for e in g.es:
                G_nx.add_edge(e.source, e.target, **e.attributes())
        return G_nx

def Draw_EPM_digraph(
    D_igraph: ig.Graph,
    title: str = "EPM Directed Graph",
    ax: Optional[plt.Axes] = None # <<< ADDED: Optional ax parameter
    ):
    """
    Draws the EPM directed graph using NetworkX and Matplotlib,
    maintaining the original style (manual layout, colors, curves).
    MODIFIED to accept an optional Axes object for external plotting control.

    Parameters:
    -----------
    D_igraph : igraph.Graph
        The EPM directed graph (igraph object) to draw.
        Expected node attributes: 'category', 'name' (for labels if needed by nx.draw).
        Expected edge attributes: 'weight'.
    title : str, optional
        The title for the plot.
    ax : matplotlib.axes.Axes, optional
        The Axes object to draw on. If None, a new figure and axes are created internally.

    Raises:
    ------
    TypeError: If input is not an igraph.Graph.
    KeyError: If required node attribute 'category' is missing.
    Exception: For other plotting related errors.
    """
    if not isinstance(D_igraph, ig.Graph):
        raise TypeError("Input must be an igraph.Graph object.")
    # Check if directed, but proceed anyway with warning if not
    if not D_igraph.is_directed():
         print("Warning: Input graph for Draw_EPM_digraph is not directed. Drawing anyway.")

    # print(f"Attempting to draw directed graph: {D_igraph.summary()}") # Uncomment for debugging

    # --- Figure and Axes Handling ---
    if ax is None:
        # If no axes provided, create figure and axes internally
        # Use dynamic figsize calculation based on node count
        num_nodes_for_size = D_igraph.vcount()
        # Ensure height scales reasonably, keep width perhaps fixed or scale differently
        dynamic_height = max(6, num_nodes_for_size * 0.5)
        fig, ax = plt.subplots(figsize=(10, dynamic_height)) # Use figsize from original logic example
        # We don't need 'created_internally' flag if we just operate on 'ax'
    else:
        # Use the provided axes object
        fig = ax.figure # Get figure from provided axes

    try:
        # --- Graph Conversion ---
        # Assumes igraph_to_networkx returns DiGraph if input is directed
        D_nx = igraph_to_networkx(D_igraph)

        if not isinstance(D_nx, nx.DiGraph):
             print("Warning: Converted graph is not a NetworkX DiGraph. Arrows might not render correctly.")

        # --- Prepare node positions and colors (Original Logic) ---
        pos_D: Dict[Any, Tuple[float, float]] = {}
        node_colors_D: List[str] = []
        node_labels_D: Dict[Any, str] = {} # For nx.draw labels
        system_nodes_D: List[Any] = []
        ancilla_nodes_D: List[Any] = []

        if not D_nx.nodes: # Handle empty graph
             print("Graph is empty, nothing to draw.")
             ax.set_axis_off() # Turn off axis even for empty graph
             # If ax was created internally, maybe close fig? Or let caller handle.
             # For now, just return as drawing is done.
             return

        # Check for necessary 'category' attribute on first node (if exists)
        first_node = list(D_nx.nodes())[0]
        if 'category' not in D_nx.nodes[first_node]:
             # Raise error as specified in original docstring attempt
             raise KeyError("Node attribute 'category' is required for positioning and coloring.")

        for node, data in D_nx.nodes(data=True):
            category = data.get('category', 'unknown')
            # Use 'name' attribute for labels if it exists, otherwise node index/id
            node_labels_D[node] = str(data.get('name', node))
            if category == 'system_nodes':
                system_nodes_D.append(node)
                node_colors_D.append('lightblue')
            elif category == 'ancilla_nodes':
                ancilla_nodes_D.append(node)
                node_colors_D.append('lightcoral')
            else:
                # Group unknown with ancilla for positioning, as per original logic
                print(f"Warning: Node '{node}' has unknown category '{category}'. Assigning default color/position.")
                ancilla_nodes_D.append(node)
                node_colors_D.append('grey')

        # --- Assign positions (Original Logic: Manual Layout) ---
        # Sort nodes before assigning positions for consistency
        # Ensure sorting works (e.g., nodes are comparable like integers or strings)
        try:
             system_nodes_D.sort()
             ancilla_nodes_D.sort()
        except TypeError:
             print("Warning: Could not sort nodes for positioning. Layout might be inconsistent.")
        # Place system nodes at x=0, ancilla nodes at x=1 vertically stacked
        pos_D.update((node, (0, -index)) for index, node in enumerate(system_nodes_D))
        pos_D.update((node, (1, -index)) for index, node in enumerate(ancilla_nodes_D))

        # --- Prepare edge colors (Original Logic) ---
        edge_colors_D: List[str] = []
        if D_nx.number_of_edges() > 0:
             # Check if weight attribute exists on the first edge
             first_edge_tuple = list(D_nx.edges())[0]
             first_edge_data = D_nx.get_edge_data(*first_edge_tuple)
             if first_edge_data is None or 'weight' not in first_edge_data:
                  print("Warning: Edge attribute 'weight' not found. Using default edge color 'grey'.")
                  edge_colors_D = ['grey'] * D_nx.number_of_edges()
             else:
                  for u, v, data in D_nx.edges(data=True):
                      weight = data.get('weight', 0) # Default to 0 if missing on specific edge
                      # Use isclose for float comparison
                      if isclose(weight, 1.0): edge_colors_D.append('red')
                      elif isclose(weight, 2.0): edge_colors_D.append('blue')
                      elif isclose(weight, 3.0): edge_colors_D.append('black')
                      else: edge_colors_D.append('grey')
        # else: edge_colors_D remains empty list [] if no edges

        # --- Draw the directed graph (Original Styling) ---
        arc_rad = 0.2 # Original curvature factor

        # --- Use ax object for drawing ---
        nx.draw(
            D_nx,
            pos=pos_D,                # Use calculated positions
            ax=ax,                    # <<< Draw on the specified axes
            labels=node_labels_D,     # Use prepared labels
            with_labels=True,         # Explicitly enable labels
            node_color=node_colors_D,
            edge_color=edge_colors_D,
            node_size=700,            # Original node size
            font_size=10,             # Original font size
            width=1.5,                # Original edge width
            arrows=True,              # Draw arrows for directed edges
            arrowsize=15,             # Original arrow size
            connectionstyle=f'arc3,rad={arc_rad}' # Apply curvature
        )

        # --- Use ax object for title and axis ---
        ax.set_title(title)           # <<< Set title on the axes
        ax.set_axis_off()             # <<< Turn off axis on the axes

        # --- Remove plt.show() ---
        # The calling script should handle showing or saving the figure

    except KeyError as e:
        print(f"Error drawing directed graph: Missing required attribute {e}")
        if ax: ax.set_axis_off() # Turn off axis even on error to clean up plot area
        # Depending on how the calling loop handles errors, re-raising might be needed
        # raise
    except Exception as e:
        print(f"An unexpected error occurred during directed graph drawing: {e}")
        if ax: ax.set_axis_off()
        # raise

# Note: Ensure 'igraph_to_networkx' function is correctly defined or imported.
# Make sure the node attributes ('category', 'name') and edge attribute ('weight')
# exist in your igraph.Graph objects passed to this function.