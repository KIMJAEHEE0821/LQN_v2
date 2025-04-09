# main_script.py
"""
Main execution script for the EPM analysis pipeline.

Handles command-line arguments, imports and calls the main pipeline function
from pipeline.py, and saves the final results to a pickle file.
"""

import argparse
import pickle
import os
import time # For overall script timing (optional)
from typing import Dict, List, Any

# --- Import the main pipeline function ---
try:
    # Assuming the pipeline logic is in pipeline.py
    from pipeline import epm_pipeline
except ImportError as e:
    print(f"Error importing the 'epm_pipeline' function from pipeline.py: {e}")
    print("Ensure pipeline.py exists and is in the Python path.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit(1)

# --- Main Execution Logic ---

def main(num_system: int, num_ancilla: int, graph_type: int, output_dir: str):
    """
    Main function to run the EPM process (via imported pipeline) and save results.

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
    script_start_time = time.time()

    # --- Run the imported EPM processing pipeline ---
    # The number of processes for parallel steps is handled within epm_pipeline
    pipeline_results = epm_pipeline(num_system, num_ancilla)
    # --- End EPM processing ---

    # Check if the pipeline produced any results
    if not pipeline_results:
        print("EPM pipeline did not produce any results to save.")
        print("Exiting.")
        script_end_time = time.time()
        print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds.")
        print("=" * 50)
        return # Exit if no results

    # --- Save Results ---
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define the output filename
        output_filename = os.path.join(output_dir, f'bigraph_result_sys{num_system}_anc{num_ancilla}_type{graph_type}.pkl')

        print(f"\nSaving final unique graph results to: '{output_filename}'...")
        save_start_time = time.time()
        with open(output_filename, 'wb') as f:
            # Use a higher pickle protocol for potentially better performance/efficiency
            pickle.dump(pipeline_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Results saved successfully in {time.time() - save_start_time:.2f} seconds.")

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

    script_end_time = time.time()
    print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds.")
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
    # Note: --processes argument is removed as process count is handled within pipeline.py

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
    # Add validation for output_dir if needed

    if error_flag:
        exit(1) # Exit if validation fails
    # --- End Input Validation ---

    # Run the main function with validated arguments
    main(args.num_system, args.num_ancilla, args.type, args.output_dir)

