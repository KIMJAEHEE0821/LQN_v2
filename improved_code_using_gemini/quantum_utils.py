# quantum_utils.py
"""
Utility functions specifically related to quantum state representation and processing.
Includes functions for converting between different state formats (e.g., Counter to SymPy).
"""

import sympy as sp
from sympy.physics.quantum import Ket
from sympy import Add, Mul # For constructing SymPy expressions
from collections import Counter, defaultdict
from itertools import groupby
from typing import List, Tuple, Dict, Any, Counter as CounterType, Union # For type hinting

def counter_to_quantum_state(counter: CounterType[str]) -> Union[Add, Mul, Ket, int]:
    """
    Converts a Counter representing quantum state counts into a SymPy quantum state expression.

    Example: Counter({'01': 2, '10': 1}) -> 2*Ket('01') + Ket('10')

    Parameters:
    -----------
    counter : CounterType[str]
        A Counter where keys are bit strings (representing basis states) and
        values are their integer coefficients (counts).

    Returns:
    --------
    Union[Add, Mul, Ket, int]
        A SymPy expression representing the unnormalized quantum state.
        - Returns Add for superpositions.
        - Returns Mul for single states with coefficient > 1.
        - Returns Ket for single states with coefficient == 1.
        - Returns 0 if the counter is empty.
    """
    if not isinstance(counter, Counter):
        raise TypeError("Input must be a collections.Counter object.")
    if not counter:
        return 0 # Return SymPy zero for an empty state

    quantum_state_terms: List[Any] = []

    # Iterate over the sorted items to ensure consistent expression order
    for state_str, coeff in sorted(counter.items()):
        if not isinstance(state_str, str):
            print(f"Warning: Key '{state_str}' is not a string. Skipping.")
            continue
        if not isinstance(coeff, int) or coeff < 0:
            print(f"Warning: Coefficient '{coeff}' for state '{state_str}' is not a non-negative integer. Skipping.")
            continue
        if coeff == 0:
            continue # Skip terms with zero coefficient

        # Create a Ket object for the basis state
        ket = Ket(state_str)

        # Multiply by the coefficient if it's not 1
        if coeff == 1:
            term = ket
        else:
            term = sp.Integer(coeff) * ket # Use sp.Integer for explicit SymPy type

        quantum_state_terms.append(term)

    # Sum the terms to get the final quantum state expression
    if not quantum_state_terms:
        return 0 # Return 0 if all terms were skipped
    elif len(quantum_state_terms) == 1:
        return quantum_state_terms[0] # Return single term directly
    else:
        # Use Add for sums of terms
        return Add(*quantum_state_terms)


def add_qc(unique_states_result: Dict[Any, List[List[Any]]]) -> Dict[Any, List[List[Any]]]:
    """
    Adds a SymPy quantum state representation to each entry in the result dictionary.

    Iterates through a dictionary where values are lists of state entries
    (e.g., [Counter, matchings, graph, index]). If an entry contains a Counter
    at index 0, it converts it to a SymPy quantum state using
    `counter_to_quantum_state` and appends it to the inner list.

    Modifies the input dictionary in-place.

    Parameters:
    -----------
    unique_states_result : Dict[Any, List[List[Any]]]
        The dictionary containing lists of state entries.
        Expected inner list format: [Counter, ...].

    Returns:
    --------
    Dict[Any, List[List[Any]]]
        The modified dictionary with the SymPy quantum state appended to each valid entry.
    """
    print("Adding SymPy quantum state representation to results...")
    processed_count = 0
    error_count = 0
    for key, value_list in unique_states_result.items():
        for entry_list in value_list:
            # Check if entry_list has at least one element and the first is a Counter
            if entry_list and isinstance(entry_list[0], Counter):
                try:
                    # Convert the Counter (at index 0) to SymPy state
                    quantum_state_sympy = counter_to_quantum_state(entry_list[0])
                    # Append the SymPy state to the end of the inner list
                    # Ensure not to append duplicates if run multiple times
                    if len(entry_list) == 4: # Assuming original format [Counter, matchings, graph, index]
                         entry_list.append(quantum_state_sympy)
                         processed_count += 1
                    elif len(entry_list) == 5 and not isinstance(entry_list[4], (Add, Mul, Ket, int)):
                         # If list has 5 elements but last isn't a Sympy state, overwrite? Or append?
                         # Let's assume we append, potentially creating more elements. Be cautious.
                         print(f"Warning: Entry list for key {key} already has 5 elements. Appending SymPy state.")
                         entry_list.append(quantum_state_sympy)
                         processed_count += 1
                    elif len(entry_list) == 5:
                         # Already has 5 elements, potentially already processed. Skip.
                         pass


                except Exception as e:
                    error_count += 1
                    print(f"Error converting counter to SymPy state for key '{key}': {e}")
            # else: # Optional: Log entries that don't match expected format
            #     print(f"Warning: Entry in key '{key}' does not start with a Counter. Skipping SymPy conversion for this entry.")

    print(f"SymPy state addition complete. Processed {processed_count} entries. Encountered {error_count} errors.")
    return unique_states_result



