# pipeline.py
"""
Defines the main EPM analysis pipeline orchestration function and its helpers,
including parallel processing steps.
"""

import os
import time
import multiprocessing
from typing import Dict, List, Any, Tuple, Optional

# --- Import functions from other analysis/generation modules ---
# Ensure these modules (graph_generation, graph_analysis) exist and are accessible
try:
    from graph_generation import EPM_bipartite_graph_generator_igraph
    from graph_analysis import (
        process_and_group_by_canonical_form,
        filter_groups_by_scc_igraph,
        # Function used by the helper for parallel processing:
        extract_unique_bigraphs_with_weights_igraph
    )
    # Import other necessary functions if the pipeline expands
    # from quantum_state_analysis import ...
    # from utils import ...
except ImportError as e:
    print(f"Error importing necessary functions in pipeline.py: {e}")
    print("Ensure graph_generation.py and graph_analysis.py are in the Python path.")
    # Optionally raise the error or exit
    raise

# --- Helper function for parallel processing ---
def process_group_for_uniqueness(group_item: Tuple[str, List[Any]]) -> Tuple[str, List[Any]]:
    """
    Helper function to process one graph group for parallel execution.
    Extracts unique graphs based on weighted isomorphism within the group.

    Parameters:
    -----------
    group_item : Tuple[str, List[Any]]
        A tuple containing (hash_key, list_of_graphs_in_group).
        The list contains igraph.Graph objects.

    Returns:
    --------
    Tuple[str, List[Any]]
        A tuple containing (hash_key, list_of_unique_weighted_graphs).
        Returns (key, []) if an error occurs during processing.
    """
    key, graph_list = group_item
    current_process_id = os.getpid() # Get current process ID (for debugging)
    # print(f"[Process {current_process_id}] Processing group {key} with {len(graph_list)} graphs...") # Debug message

    try:
        # Call the function to extract unique graphs based on weights
        # This function should be imported from graph_analysis.py
        unique_graphs = extract_unique_bigraphs_with_weights_igraph(graph_list)
        # print(f"[Process {current_process_id}] Finished group {key}, found {len(unique_graphs)} unique graphs.") # Debug message
        return key, unique_graphs
    except Exception as e:
        # Log error during individual group processing and return empty result for this group
        print(f"Error processing group {key} in process {current_process_id}: {e}")
        # import traceback # Uncomment for detailed traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        return key, []


# --- Main EPM Pipeline Orchestration Function ---
def epm_pipeline(num_system: int, num_ancilla: int, num_processes: Optional[int] = None) -> Dict[str, List[Any]]:
    """
    Orchestrates the main EPM pipeline steps: generation, grouping, filtering,
    and parallel extraction of unique weighted graphs.

    Parameters:
    -----------
    num_system : int
        Number of system nodes.
    num_ancilla : int
        Number of ancilla nodes.
    num_processes : Optional[int], optional
        Number of parallel processes to use for Step 4.
        If None, attempts to read SLURM_NPROCS or defaults to os.cpu_count().

    Returns:
    --------
    Dict[str, List[Any]]
        Final unique bipartite graph groups dictionary. Returns empty dict on error.
    """
    print("-" * 30)
    print("Starting EPM Pipeline...")
    start_time_pipeline = time.time()

    try:
        # --- Step 1: Generate initial bipartite graphs ---
        print("Step 1: Generating initial bipartite graphs...")
        step_start_time = time.time()
        # Ensure EPM_bipartite_graph_generator_igraph is imported
        graph_generator = EPM_bipartite_graph_generator_igraph(num_system, num_ancilla)
        print(f"Step 1 finished in {time.time() - step_start_time:.2f} seconds.")

        # --- Step 2: Group graphs by structural isomorphism ---
        print("Step 2: Grouping graphs by structural isomorphism...")
        step_start_time = time.time()
        # Ensure process_and_group_by_canonical_form is imported
        # This function consumes the generator from Step 1
        canonical_groups = process_and_group_by_canonical_form(graph_generator)
        print(f"Step 2 finished in {time.time() - step_start_time:.2f} seconds.")
        if not canonical_groups:
            print("No graph groups generated.")
            return {}

        # --- Step 3: Filter groups based on SCC condition ---
        print("Step 3: Filtering groups based on SCC condition...")
        step_start_time = time.time()
        # Ensure filter_groups_by_scc_igraph is imported
        scc_filtered_groups = filter_groups_by_scc_igraph(canonical_groups)
        print(f"Step 3 finished in {time.time() - step_start_time:.2f} seconds.")
        if not scc_filtered_groups:
            print("No graph groups passed the SCC filter.")
            return {}

        # --- Step 4: Parallel Extraction of Unique Weighted Graphs ---
        print(f"Step 4: Extracting unique graphs based on weighted isomorphism (using parallel processing)...")
        step_start_time = time.time()

        # Determine number of processes to use
        if num_processes is None:
            try:
                # Try getting core count from SLURM environment variable first
                slurm_procs = os.environ.get('SLURM_NPROCS') or os.environ.get('SLURM_CPUS_PER_TASK')
                if slurm_procs:
                    num_processes = int(slurm_procs)
                    print(f"Using {num_processes} processes (detected from SLURM environment variable).")
                else:
                    # Fallback to os.cpu_count() if not in SLURM or var not set
                    num_processes = os.cpu_count()
                    print(f"Using {num_processes} processes (detected from os.cpu_count()).")
            except (NotImplementedError, ValueError, TypeError):
                print("Warning: Could not detect CPU count automatically. Defaulting to 1 process.")
                num_processes = 1
        else:
             print(f"Using specified number of processes: {num_processes}")

        # Ensure at least 1 process
        num_processes = max(1, num_processes)

        # Prepare data for parallel processing
        items_to_process = list(scc_filtered_groups.items())
        unique_bigraph_groups = {} # Initialize result dictionary

        if not items_to_process:
             print("No groups to process for unique weighted graph extraction.")
        elif num_processes == 1:
             # Run sequentially if only 1 process is requested/available
             print("Running Step 4 sequentially (num_processes=1)...")
             for key, graph_list in items_to_process:
                 # Call the helper function directly
                 _, unique_graphs = process_group_for_uniqueness((key, graph_list))
                 if unique_graphs: # Only add if results are not empty
                      unique_bigraph_groups[key] = unique_graphs
        else:
            # Use multiprocessing Pool for parallel execution
            # Adjust chunksize dynamically, ensuring it's at least 1
            # A larger chunksize can reduce overhead but increase memory per worker
            chunk_size = max(1, len(items_to_process) // (num_processes * 2)) # Example chunksize logic
            print(f"Starting parallel processing with {num_processes} workers, chunksize={chunk_size}...")
            try:
                # Create a Pool of worker processes
                with multiprocessing.Pool(processes=num_processes) as pool:
                    # Map the helper function to the data items in parallel
                    # pool.map processes items in chunks and preserves order
                    parallel_results = pool.map(process_group_for_uniqueness, items_to_process, chunksize=chunk_size)

                # Collect results into the final dictionary, filtering empty ones
                unique_bigraph_groups = {key: graphs for key, graphs in parallel_results if graphs}
                print(f"Parallel extraction complete. Found results for {len(unique_bigraph_groups)} groups.")

            except Exception as parallel_error:
                 # Catch errors during the parallel map execution
                 print(f"Error during parallel processing: {parallel_error}")
                 print("Aborting pipeline.")
                 return {} # Exit on parallel processing error

        print(f"Step 4 finished in {time.time() - step_start_time:.2f} seconds.")

        if not unique_bigraph_groups:
             print("No unique weighted graphs found after final extraction.")
             # Decide whether to return {} or proceed

        # --- Pipeline 완료 ---
        end_time_pipeline = time.time()
        total_unique_count = sum(len(g_list) for g_list in unique_bigraph_groups.values())
        print(f"EPM Pipeline finished successfully in {end_time_pipeline - start_time_pipeline:.2f} seconds.")
        print(f"Total unique weighted graphs found: {total_unique_count}")
        print("-" * 30)
        return unique_bigraph_groups

    except Exception as e:
        # Catch any other unexpected errors during the pipeline setup or sequential steps
        print(f"\n--- An error occurred during the EPM pipeline ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        # import traceback # Uncomment for detailed traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        print("Pipeline aborted.")
        print("-" * 30)
        return {}