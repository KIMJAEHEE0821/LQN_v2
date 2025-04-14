# main_script.py
"""
Main execution script for the EPM (Entanglement Purification Measure) analysis pipeline.

Handles command-line arguments, orchestrates the process by calling functions
from other modules, and saves the final results (unique bipartite graphs)
to a pickle file.
"""

import argparse
import pickle
import os
import time # To measure execution time
from typing import Dict, List, Any

# --- Import functions from other modules ---
# Adjust module names if they are different in your structure
try:
    from graph_generation import EPM_bipartite_graph_generator_igraph
    from graph_analysis import (
        process_and_group_by_canonical_form,
        filter_groups_by_scc_igraph,
        extract_unique_bigraphs_from_groups_igraph
    )
    # Import quantum state analysis functions if needed later in the pipeline
    # from quantum_state_analysis import process_graph_dict
except ImportError as e:
    print(f"Error importing necessary functions: {e}")
    print("Please ensure graph_generation.py and graph_analysis.py are in the Python path.")
    exit(1)

# --- EPM Pipeline Orchestration ---

def epm_pipeline(num_system: int, num_ancilla: int) -> Dict[str, List[Any]]:
    """
    Orchestrates the main EPM pipeline steps: generation, grouping, filtering.

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
        their structural hash. The value list contains unique graphs based on
        weighted isomorphism. Returns an empty dict if errors occur or no
        graphs pass the filters.
    """
    print("-" * 30)
    print("Starting EPM Pipeline...")
    start_time = time.time()

    try:
        # Step 1: Generate initial bipartite graphs
        print("Step 1: Generating initial bipartite graphs...")
        graph_generator = EPM_bipartite_graph_generator_igraph(num_system, num_ancilla)
        # Note: graph_generator is an iterator

        # Step 2: Group graphs by weight-agnostic canonical form (structural isomorphism)
        print("Step 2: Grouping graphs by structural isomorphism...")
        canonical_groups = process_and_group_by_canonical_form(graph_generator)
        if not canonical_groups:
            print("No graph groups generated.")
            return {}

        # Step 3: Filter groups based on the SCC condition of the derived directed graph
        print("Step 3: Filtering groups based on SCC condition...")
        scc_filtered_groups = filter_groups_by_scc_igraph(canonical_groups)
        if not scc_filtered_groups:
            print("No graph groups passed the SCC filter.")
            return {}

        # Step 4: Extract unique graphs within each group based on weighted isomorphism
        print("Step 4: Extracting unique graphs based on weighted isomorphism...")
        unique_bigraph_groups = extract_unique_bigraphs_from_groups_igraph(scc_filtered_groups)
        if not unique_bigraph_groups:
             print("No unique weighted graphs found after final extraction.")
             return {}

        # Optional Step 5: Further analysis (e.g., quantum state analysis) could be called here
        # print("Step 5: Performing quantum state analysis...")
        # quantum_results = process_graph_dict(unique_bigraph_groups)
        # For now, this script focuses on saving the unique graphs as per the original main logic.

        end_time = time.time()
        print(f"EPM Pipeline finished successfully in {end_time - start_time:.2f} seconds.")
        print("-" * 30)
        # Return the unique bipartite graph groups found in Step 4
        return unique_bigraph_groups

    except Exception as e:
        print(f"\n--- An error occurred during the EPM pipeline ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        # Optionally add more detailed traceback logging here
        # import traceback
        # traceback.print_exc()
        print("Pipeline aborted.")
        print("-" * 30)
        return {} # Return empty dict on error


# --- Main Execution Logic ---

def main(num_system: int, num_ancilla: int, graph_type: int, output_dir: str):
    """
    Main function to run the EPM pipeline and save the results.

    Parameters:
    -----------
    num_system : int
        Number of system nodes.
    num_ancilla : int
        Number of ancilla nodes.
    graph_type : int
        Graph type identifier (used for filename).
    output_dir : str
        Directory where results will be saved.
    """
    print("=" * 50)
    print(f"Running EPM Analysis for sys={num_system}, anc={num_ancilla}, type={graph_type}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)

    # --- Run the main EPM processing pipeline ---
    pipeline_results = epm_pipeline(num_system, num_ancilla)
    # --- End EPM processing ---

    # Check if the pipeline produced any results
    if not pipeline_results:
        print("EPM pipeline did not produce any results to save.")
        print("Exiting.")
        return # Exit if no results

    # --- Save Results ---
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define the output filename
        # Using the 'graph_type' parameter in the filename as per original logic
        output_filename = os.path.join(output_dir, f'bigraph_result_sys{num_system}_anc{num_ancilla}_type{graph_type}.pkl')

        print(f"\nSaving final unique graph results to: '{output_filename}'...")
        with open(output_filename, 'wb') as f:
            # Save the dictionary containing the unique graph groups
            pickle.dump(pipeline_results, f)
        print("Results saved successfully.")

    except FileNotFoundError:
        print(f"Error: Could not write to file. Path might be invalid: {output_filename}")
    except PermissionError:
         print(f"Error: Permission denied when trying to write to {output_filename}")
    except pickle.PicklingError as e:
        print(f"Error: Failed to serialize the results using pickle: {e}")
    except MemoryError:
         print("Error: Insufficient memory to save the results.")
    except Exception as e:
        print(f"An unexpected error occurred during result saving: {e}")

    print("=" * 50)
    print("Script finished.")
    print("=" * 50)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run the EPM graph analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )
    parser.add_argument('--num_system', type=int, required=True,
                        help='Number of system nodes (non-negative integer)')
    parser.add_argument('--num_ancilla', type=int, required=True,
                        help='Number of ancilla nodes (non-negative integer)')
    parser.add_argument('--type', type=int, default=0,
                        help='Graph type identifier (used for output filename)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save the output pickle file')

    # Parse arguments from command line
    args = parser.parse_args()

    # --- Input Validation ---
    error_flag = False
    if args.num_system < 0:
        print("Error: --num_system must be a non-negative integer.")
        error_flag = True
    if args.num_ancilla < 0:
        print("Error: --num_ancilla must be a non-negative integer.")
        error_flag = True
    # Add validation for output_dir if needed (e.g., check if path is writable, though makedirs handles creation)

    if error_flag:
        exit(1) # Exit if validation fails
    # --- End Input Validation ---

    # Run the main function with validated arguments
    main(args.num_system, args.num_ancilla, args.type, args.output_dir)
