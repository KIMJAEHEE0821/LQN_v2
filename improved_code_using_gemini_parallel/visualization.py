# visualization.py
"""
Functions for visualizing EPM bipartite and directed graphs using
matplotlib and networkx.
"""
import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig # Needed for type hinting
from typing import Optional, Dict, List, Any, Tuple

# Import the conversion utility from utils.py
try:
    from utils import igraph_to_networkx
except ImportError:
    print("Error: Could not import 'igraph_to_networkx' from 'utils'.")
    print("Ensure utils.py is in the Python path.")
    # Define a dummy function to avoid crashing if utils is missing,
    # but visualization will likely fail.
    def igraph_to_networkx(g: ig.Graph) -> nx.Graph:
        print("Warning: Using dummy igraph_to_networkx function.")
        return nx.Graph() # Return an empty graph

# Note: igraph_to_networkx function definition assumed from previous answers
# (Make sure you have the version that creates nx.DiGraph for directed input)
# def igraph_to_networkx(g_igraph: ig.Graph) -> nx.Graph | nx.DiGraph: ...



def Draw_EPM_bipartite_graph(B_igraph: ig.Graph, title: str = "EPM Bipartite Graph"):
    """
    Draws the EPM bipartite graph using NetworkX and Matplotlib.

    Nodes are positioned based on their category ('system_nodes', 'ancilla_nodes',
    'sculpting_nodes') and colored accordingly. Edges are colored based on weight.

    Parameters:
    -----------
    B_igraph : igraph.Graph
        The EPM bipartite graph (igraph object) to draw.
        Expected node attributes: 'category', 'name'.
        Expected edge attributes: 'weight'.
    title : str, optional
        The title for the plot (default: "EPM Bipartite Graph").

    Raises:
    ------
    TypeError: If input is not an igraph.Graph.
    KeyError: If required node/edge attributes ('category', 'name', 'weight') are missing.
    Exception: For other plotting related errors.
    """
    if not isinstance(B_igraph, ig.Graph):
        raise TypeError("Input must be an igraph.Graph object.")

    print(f"Attempting to draw bipartite graph: {B_igraph.summary()}")

    try:
        # Convert igraph to NetworkX graph using the utility function
        B_nx = igraph_to_networkx(B_igraph)

        # --- Prepare node positions and colors ---
        pos: Dict[Any, Tuple[float, float]] = {}
        node_colors: List[str] = []
        system_nodes: List[Any] = []
        ancilla_nodes: List[Any] = []
        sculpting_nodes: List[Any] = []

        # Check for necessary 'category' attribute and classify nodes
        if 'category' not in B_nx.nodes[list(B_nx.nodes())[0]]:
             raise KeyError("Node attribute 'category' is required for positioning and coloring.")

        for node, data in B_nx.nodes(data=True):
            category = data.get('category', 'unknown') # Default if missing
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
                # Handle unknown categories if necessary
                print(f"Warning: Node '{node}' has unknown category '{category}'. Assigning default color.")
                sculpting_nodes.append(node) # Treat as sculpting for positioning?
                node_colors.append('grey')

        # Assign positions - Place system/ancilla on left, sculpting on right
        # Sort nodes for consistent vertical ordering
        system_nodes.sort()
        ancilla_nodes.sort()
        sculpting_nodes.sort() # Sort sculpting nodes, assuming names are comparable (e.g., '0', '1', ...)

        y_pos_sys = {node: -index for index, node in enumerate(system_nodes)}
        y_pos_anc = {node: -(index + len(system_nodes)) for index, node in enumerate(ancilla_nodes)}
        # Try to align sculpting nodes vertically with their typical counterparts if possible
        # This simple alignment assumes sculpting nodes '0'...'N-1' correspond to sys+anc nodes
        num_left_nodes = len(system_nodes) + len(ancilla_nodes)
        y_pos_sculpt = {}
        current_y = 0
        for node in sculpting_nodes:
             # Simple vertical stacking for sculpting nodes
             y_pos_sculpt[node] = current_y
             current_y -= 1 # Decrement y for next node

        # Combine positions
        pos.update((node, (0, y_pos_sys[node])) for node in system_nodes)
        pos.update((node, (0, y_pos_anc[node])) for node in ancilla_nodes)
        # Position sculpting nodes on the right (x=1)
        pos.update((node, (1, y_pos_sculpt.get(node, 0))) for node in sculpting_nodes) # Use .get for safety

        # --- Prepare edge colors ---
        edge_colors: List[str] = []
        # Check for necessary 'weight' attribute
        if B_nx.number_of_edges() > 0:
             # Check attribute on the first edge
             first_edge_data = B_nx.get_edge_data(*list(B_nx.edges())[0])
             if 'weight' not in first_edge_data:
                  print("Warning: Edge attribute 'weight' not found. Using default edge color.")
                  edge_colors = ['black'] * B_nx.number_of_edges()
             else:
                  for u, v, data in B_nx.edges(data=True):
                      weight = data.get('weight', 0) # Default if missing on specific edge
                      if weight == 1.0:
                          edge_colors.append('red')   # System 'red' connection
                      elif weight == 2.0:
                          edge_colors.append('blue')  # System 'blue' connection
                      elif weight == 3.0:
                          edge_colors.append('black') # Ancilla connection
                      else:
                          # print(f"Warning: Edge ({u},{v}) has unrecognized weight {weight}. Using default color.")
                          edge_colors.append('grey') # Default for other weights
        else:
             edge_colors = [] # No edges


        # --- Draw the graph ---
        plt.figure(figsize=(8, max(6, B_nx.number_of_nodes() * 0.5))) # Adjust figure size dynamically

        nx.draw(
            B_nx,
            pos,
            with_labels=True,     # Show node names (ensure 'name' attribute exists)
            node_color=node_colors,
            edge_color=edge_colors,
            node_size=700,        # Adjust node size
            font_size=10,         # Adjust font size
            width=1.5             # Edge width
        )
        plt.title(title)
        plt.axis('off') # Turn off axis
        #plt.show()

    except KeyError as e:
        print(f"Error drawing bipartite graph: Missing required attribute {e}")
        # Optionally re-raise or handle differently
    except Exception as e:
        print(f"An unexpected error occurred during bipartite graph drawing: {e}")
        # Optionally re-raise


# def Draw_EPM_digraph(D_igraph: ig.Graph, title: str = "EPM Directed Graph"):
#     """
#     Draws the EPM directed graph using NetworkX and Matplotlib.

#     Nodes are positioned based on their category ('system_nodes', 'ancilla_nodes')
#     and colored accordingly. Edges are colored based on weight and drawn with arrows.
#     Uses curved edges for better visualization.

#     Parameters:
#     -----------
#     D_igraph : igraph.Graph
#         The EPM directed graph (igraph object) to draw.
#         Expected node attributes: 'category', 'name'.
#         Expected edge attributes: 'weight'.
#     title : str, optional
#         The title for the plot (default: "EPM Directed Graph").

#     Raises:
#     ------
#     TypeError: If input is not an igraph.Graph.
#     KeyError: If required node/edge attributes ('category', 'name', 'weight') are missing.
#     Exception: For other plotting related errors.
#     """
#     if not isinstance(D_igraph, ig.Graph):
#         raise TypeError("Input must be an igraph.Graph object.")
#     if not D_igraph.is_directed():
#          print("Warning: Input graph for Draw_EPM_digraph is not directed. Drawing anyway.")

#     print(f"Attempting to draw directed graph: {D_igraph.summary()}")

#     try:
#         # Convert igraph to NetworkX graph
#         D_nx = igraph_to_networkx(D_igraph) # Should return DiGraph if input is directed

#         if not isinstance(D_nx, nx.DiGraph):
#              print("Warning: Converted graph is not a NetworkX DiGraph. Arrows might not render correctly.")


#         # --- Prepare node positions and colors ---
#         pos_D: Dict[Any, Tuple[float, float]] = {}
#         node_colors_D: List[str] = []
#         system_nodes_D: List[Any] = []
#         ancilla_nodes_D: List[Any] = []

#         # Check for necessary 'category' attribute
#         if not D_nx.nodes: # Handle empty graph
#              print("Graph is empty, nothing to draw.")
#              return
#         if 'category' not in D_nx.nodes[list(D_nx.nodes())[0]]:
#              raise KeyError("Node attribute 'category' is required for positioning and coloring.")

#         for node, data in D_nx.nodes(data=True):
#             category = data.get('category', 'unknown')
#             if category == 'system_nodes':
#                 system_nodes_D.append(node)
#                 node_colors_D.append('lightblue')
#             elif category == 'ancilla_nodes':
#                 ancilla_nodes_D.append(node)
#                 node_colors_D.append('lightcoral')
#             else:
#                 print(f"Warning: Node '{node}' has unknown category '{category}'. Assigning default color.")
#                 # Decide how to position unknown nodes, e.g., group with ancillas
#                 ancilla_nodes_D.append(node)
#                 node_colors_D.append('grey')

#         # Assign positions - Simple vertical layout for now
#         system_nodes_D.sort()
#         ancilla_nodes_D.sort()
#         # Place system nodes at x=0, ancilla nodes at x=1 (or adjust as needed)
#         pos_D.update((node, (0, -index)) for index, node in enumerate(system_nodes_D))
#         pos_D.update((node, (1, -index)) for index, node in enumerate(ancilla_nodes_D))

#         # --- Prepare edge colors ---
#         edge_colors_D: List[str] = []
#         # Check for necessary 'weight' attribute
#         if D_nx.number_of_edges() > 0:
#              first_edge_data = D_nx.get_edge_data(*list(D_nx.edges())[0])
#              if 'weight' not in first_edge_data:
#                   print("Warning: Edge attribute 'weight' not found. Using default edge color.")
#                   edge_colors_D = ['black'] * D_nx.number_of_edges()
#              else:
#                   for u, v, data in D_nx.edges(data=True):
#                       weight = data.get('weight', 0)
#                       # Original code compared weights to strings '0', '1' - let's assume float/int
#                       if weight == 1.0: # Corresponds to '0' state? (Red in bipartite)
#                           edge_colors_D.append('red')
#                       elif weight == 2.0: # Corresponds to '1' state? (Blue in bipartite)
#                           edge_colors_D.append('blue')
#                       elif weight == 3.0: # Corresponds to '2' state? (Black in bipartite)
#                           edge_colors_D.append('black')
#                       else:
#                           # print(f"Warning: Edge ({u},{v}) has unrecognized weight {weight}. Using default color.")
#                           edge_colors_D.append('grey')
#         else:
#             edge_colors_D = []


#         # --- Draw the directed graph with curved edges ---
#         plt.figure(figsize=(8, max(6, D_nx.number_of_nodes() * 0.5)))

#         # Use connectionstyle for curved edges
#         # Increase arc_rad for more curve, decrease for less
#         arc_rad = 0.2

#         nx.draw(
#             D_nx,
#             pos_D,
#             with_labels=True,
#             node_color=node_colors_D,
#             edge_color=edge_colors_D,
#             node_size=700,
#             font_size=10,
#             width=1.5,
#             arrows=True,          # Draw arrows for directed edges
#             arrowsize=15,         # Adjust arrow size
#             connectionstyle=f'arc3,rad={arc_rad}' # Apply curvature
#         )

#         plt.title(title)
#         plt.axis('off')
#         plt.show()

#     except KeyError as e:
#         print(f"Error drawing directed graph: Missing required attribute {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred during directed graph drawing: {e}")


import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np # Keep for dynamic figsize calculation
from typing import Dict, Any, Tuple, List, Optional

# Note: Assumes the existence of 'igraph_to_networkx' function from previous context
# def igraph_to_networkx(g_igraph: ig.Graph) -> nx.Graph | nx.DiGraph: ...

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
        Expected node attributes: 'category', 'name'.
        Expected edge attributes: 'weight'.
    title : str, optional
        The title for the plot.
    ax : matplotlib.axes.Axes, optional
        The Axes object to draw on. If None, a new figure and axes are created internally.

    Raises:
    ------
    TypeError: If input is not an igraph.Graph.
    KeyError: If required node/edge attributes ('category', 'name', 'weight') are missing.
    Exception: For other plotting related errors.
    """
    if not isinstance(D_igraph, ig.Graph):
        raise TypeError("Input must be an igraph.Graph object.")
    if not D_igraph.is_directed():
         print("Warning: Input graph for Draw_EPM_digraph is not directed. Drawing anyway.")

    # print(f"Attempting to draw directed graph: {D_igraph.summary()}") # Uncomment for debugging

    # --- Figure and Axes Handling ---
    if ax is None:
        # If no axes provided, create figure and axes internally
        # Use dynamic figsize calculation from the original code
        num_nodes_for_size = D_igraph.vcount()
        dynamic_height = max(6, num_nodes_for_size * 0.5) # Original height calculation
        fig, ax = plt.subplots(figsize=(8, dynamic_height)) # Original width was fixed at 8
        created_internally = True
    else:
        # Use the provided axes object
        fig = ax.figure # Get figure from provided axes
        created_internally = False

    try:
        # --- Graph Conversion ---
        D_nx = igraph_to_networkx(D_igraph) # Assumes DiGraph if input is directed

        if not isinstance(D_nx, nx.DiGraph):
             print("Warning: Converted graph is not a NetworkX DiGraph. Arrows might not render correctly.")

        # --- Prepare node positions and colors (Original Logic) ---
        pos_D: Dict[Any, Tuple[float, float]] = {}
        node_colors_D: List[str] = []
        system_nodes_D: List[Any] = []
        ancilla_nodes_D: List[Any] = []

        if not D_nx.nodes:
             print("Graph is empty, nothing to draw.")
             ax.set_axis_off() # Turn off axis even for empty graph
             return
        if not D_nx.nodes or 'category' not in D_nx.nodes[list(D_nx.nodes())[0]]:
             # Handle missing category - raise error as per original docstring
             raise KeyError("Node attribute 'category' is required for positioning and coloring.")

        for node, data in D_nx.nodes(data=True):
            category = data.get('category', 'unknown')
            if category == 'system_nodes':
                system_nodes_D.append(node)
                node_colors_D.append('lightblue')
            elif category == 'ancilla_nodes':
                ancilla_nodes_D.append(node)
                node_colors_D.append('lightcoral')
            else:
                # Original code groups unknown with ancilla for positioning
                print(f"Warning: Node '{node}' has unknown category '{category}'. Assigning default color.")
                ancilla_nodes_D.append(node)
                node_colors_D.append('grey')

        # --- Assign positions (Original Logic: Manual Layout) ---
        system_nodes_D.sort()
        ancilla_nodes_D.sort()
        # Place system nodes at x=0, ancilla nodes at x=1
        pos_D.update((node, (0, -index)) for index, node in enumerate(system_nodes_D))
        pos_D.update((node, (1, -index)) for index, node in enumerate(ancilla_nodes_D))

        # --- Prepare edge colors (Original Logic) ---
        edge_colors_D: List[str] = []
        if D_nx.number_of_edges() > 0:
             first_edge_data = D_nx.get_edge_data(*list(D_nx.edges())[0])
             # Check if weight attribute exists
             if first_edge_data is None or 'weight' not in first_edge_data:
                  print("Warning: Edge attribute 'weight' not found. Using default edge color.")
                  edge_colors_D = ['black'] * D_nx.number_of_edges()
             else:
                  for u, v, data in D_nx.edges(data=True):
                      weight = data.get('weight', 0)
                      if weight == 1.0: edge_colors_D.append('red')
                      elif weight == 2.0: edge_colors_D.append('blue')
                      elif weight == 3.0: edge_colors_D.append('black')
                      else: edge_colors_D.append('grey')
        # else: edge_colors_D remains empty list [] if no edges

        # --- Draw the directed graph (Original Styling) ---
        arc_rad = 0.2 # Original curvature

        # --- Use ax object for drawing ---
        nx.draw(
            D_nx,
            pos=pos_D,                # Original manual layout
            ax=ax,                    # <<< Draw on the specified axes
            with_labels=True,
            node_color=node_colors_D,
            edge_color=edge_colors_D,
            node_size=700,            # Original node size
            font_size=10,             # Original font size
            width=1.5,                # Original edge width
            arrows=True,              # Original arrows
            arrowsize=15,             # Original arrow size
            connectionstyle=f'arc3,rad={arc_rad}' # Original curvature
        )

        # --- Use ax object for title and axis ---
        ax.set_title(title)           # <<< Set title on the axes
        ax.set_axis_off()             # <<< Turn off axis on the axes

        # --- Remove plt.show() ---
        # The calling script will handle showing or saving the figure

    except KeyError as e:
        print(f"Error drawing directed graph: Missing required attribute {e}")
        if ax: ax.set_axis_off() # Turn off axis even on error
        # Re-raise might be appropriate depending on how the loop handles errors
        # raise
    except Exception as e:
        print(f"An unexpected error occurred during directed graph drawing: {e}")
        if ax: ax.set_axis_off() # Turn off axis even on error
        # raise