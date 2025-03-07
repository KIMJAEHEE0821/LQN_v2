import argparse
import time
import pickle
import multiprocessing

# Import functions from the module
from epm_parallel_functions import epm_process_parallel
# Import original functions
from epm_functions import EPM_bipartite_graph_generator_igraph  # Assuming original functions are in this module

def main(num_system, num_ancilla, n_workers=None, type="default"):
    """
    Main function to execute the parallel EPM process
    
    Parameters:
    -----------
    num_system : int
        Number of system nodes
    num_ancilla : int
        Number of ancilla nodes
    n_workers : int, optional
        Number of worker processes to use
    type : str
        Graph type
    """
    start_time = time.time()
    
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    print(f"=== Starting EPM Process ===")
    print(f"Number of system nodes: {num_system}")
    print(f"Number of ancilla nodes: {num_ancilla}")
    print(f"Number of parallel workers: {n_workers}")
    print(f"Graph type: {type}")
    
    # Execute EPM process with parallel processing
    unique_bigraph = epm_process_parallel(num_system, num_ancilla, n_workers)
    
    # Save results to file
    bigraph_filename = f'bigraph_result_sys{num_system}_anc{num_ancilla}_type{type}.pkl'
    
    # Calculate number of graphs
    total_graphs = sum(len(graphs) for graphs in unique_bigraph.values())
    
    print(f"Found {len(unique_bigraph)} groups with {total_graphs} unique graphs")
    
    with open(bigraph_filename, 'wb') as f:
        pickle.dump(unique_bigraph, f)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Results saved to '{bigraph_filename}'")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    return unique_bigraph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EPM graphs with parallel processing")
    parser.add_argument('--num_system', type=int, required=True, help='Number of system nodes')
    parser.add_argument('--num_ancilla', type=int, required=True, help='Number of ancilla nodes')
    parser.add_argument('--n_workers', type=int, default=None, help='Number of worker processes (default: all available cores)')
    parser.add_argument('--type', type=str, default="default", help='Graph type')
    
    args = parser.parse_args()
    main(args.num_system, args.num_ancilla, args.n_workers, args.type)