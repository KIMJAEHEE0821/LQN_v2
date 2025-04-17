# pipeline.py
"""
Defines the main EPM analysis pipeline orchestration function, based on the
original sequential processing logic.
"""

import time
from typing import Dict, List, Any

# --- Import functions from other analysis/generation modules ---
# Ensure these modules exist and are accessible
try:
    # Adjust module names if they are different in your structure
    from graph_generation import EPM_bipartite_graph_generator_igraph
    from graph_analysis import (
        process_and_group_by_canonical_form,
        filter_groups_by_scc_igraph,
        extract_unique_bigraphs_from_groups_igraph # Make sure this name is correct
    )
    # Import quantum state analysis functions if needed later in the pipeline
    # from quantum_state_analysis import process_graph_dict
except ImportError as e:
    print(f"Error importing necessary functions in pipeline.py: {e}")
    print("Please ensure graph_generation.py and graph_analysis.py are in the Python path.")
    raise # Reraise the error to stop execution if imports fail

# --- EPM Pipeline Orchestration Function (Original Sequential Logic) ---

def epm_pipeline(num_system: int, num_ancilla: int) -> Dict[str, List[Any]]:
    """
    Orchestrates the main EPM pipeline steps: generation, grouping, filtering.
    This version uses the original sequential logic.

    Parameters:
    -----------
    num_system : int
        Number of system nodes.
    num_ancilla : int
        Number of ancilla nodes.

    Returns:
    --------
    Dict[str, List[Any]]
        A dictionary containing the final unique bipartite graphs, grouped by
        their structural hash. Returns an empty dict if errors occur or no
        graphs pass the filters.
    """
    print("-" * 30)
    print("Starting EPM Pipeline (Sequential Version)...")
    start_time_pipeline = time.time()

    try:
        # Step 1: Generate initial bipartite graphs
        print("Step 1: Generating initial bipartite graphs...")
        step_start_time = time.time()
        graph_generator = EPM_bipartite_graph_generator_igraph(num_system, num_ancilla)
        print(f"Step 1 finished in {time.time() - step_start_time:.2f} seconds.")
        # Note: graph_generator is an iterator

        # Step 2: Group graphs by weight-agnostic canonical form (structural isomorphism)
        print("Step 2: Grouping graphs by structural isomorphism...")
        step_start_time = time.time()
        canonical_groups = process_and_group_by_canonical_form(graph_generator)
        print(f"Step 2 finished in {time.time() - step_start_time:.2f} seconds.")
        if not canonical_groups:
            print("No graph groups generated.")
            return {}

        # Step 3: Filter groups based on the SCC condition of the derived directed graph
        print("Step 3: Filtering groups based on SCC condition...")
        step_start_time = time.time()
        scc_filtered_groups = filter_groups_by_scc_igraph(canonical_groups)
        print(f"Step 3 finished in {time.time() - step_start_time:.2f} seconds.")
        if not scc_filtered_groups:
            print("No graph groups passed the SCC filter.")
            return {}

        # Step 4: Extract unique graphs within each group based on weighted isomorphism
        # This step uses the sequential function from graph_analysis
        print("Step 4: Extracting unique graphs based on weighted isomorphism (sequentially)...")
        step_start_time = time.time()
        # Ensure the correct function name is used here
        unique_bigraph_groups = extract_unique_bigraphs_from_groups_igraph(scc_filtered_groups)
        print(f"Step 4 finished in {time.time() - step_start_time:.2f} seconds.")
        if not unique_bigraph_groups:
             print("No unique weighted graphs found after final extraction.")
             return {}

        # --- Pipeline 완료 ---
        end_time_pipeline = time.time()
        total_unique_count = sum(len(g_list) for g_list in unique_bigraph_groups.values())
        print(f"EPM Pipeline finished successfully in {end_time_pipeline - start_time_pipeline:.2f} seconds.")
        print(f"Total unique weighted graphs found: {total_unique_count}")
        print("-" * 30)
        # Return the unique bipartite graph groups found in Step 4
        return unique_bigraph_groups

    except Exception as e:
        print(f"\n--- An error occurred during the EPM pipeline ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        # import traceback # Uncomment for detailed traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        print("Pipeline aborted.")
        print("-" * 30)
        return {} # Return empty dict on error