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


def convert_to_quantum_notation(
    list_of_state_counters: List[CounterType[str]],
    method: str = 'sympy'
    ) -> List[Tuple[Union[str, Add, Mul, Ket, int], CounterType[str]]]:
    """
    Converts a list of state Counters to a list of quantum state expressions.

    DEPRECATED? This seems less useful than `counter_to_quantum_state` and `add_qc`.
    It takes a list of Counters and returns expressions, but doesn't integrate
    with the main data structure like `add_qc`. Keeping for reference if used elsewhere.

    Parameters:
    -----------
    list_of_state_counters : List[CounterType[str]]
        A list of Counters, each representing a quantum state.
    method : str, optional
        Method for conversion: 'sympy' (default) or 'string'.

    Returns:
    --------
    List[Tuple[Union[str, Add, Mul, Ket, int], CounterType[str]]]
        A list of tuples, where each tuple contains:
        (quantum state expression (SymPy or string), original Counter).
    """
    print("Warning: `convert_to_quantum_notation` might be deprecated. Consider using `add_qc`.")
    all_expressions: List[Tuple[Union[str, Add, Mul, Ket, int], CounterType[str]]] = []

    for counter in list_of_state_counters:
        if not isinstance(counter, Counter):
            print(f"Warning: Item is not a Counter: {type(counter)}. Skipping.")
            continue

        expression: Union[str, Add, Mul, Ket, int]
        if method == 'sympy':
            expression = counter_to_quantum_state(counter)
        elif method == 'string':
            # String-based method (simple superposition string)
            terms = []
            for state, coeff in sorted(counter.items()):
                 if coeff == 1:
                      terms.append(f'|{state}⟩')
                 elif coeff > 1:
                      terms.append(f'{coeff}|{state}⟩')
            expression = ' + '.join(terms) if terms else '0'
        else:
            raise ValueError("Method must be either 'string' or 'sympy'.")

        all_expressions.append((expression, counter))

    return all_expressions


def qs_generator(
    processed_results: Dict[str, List[List[Any]]],
    methods: str = 'sympy'
    ) -> Dict[str, List[Tuple[Union[str, Add, Mul, Ket, int], List[List[Any]]]]]:
    """
    Generates quantum state expressions for processed results.

    DEPRECATED? This seems overly complex and likely superseded by `add_qc`.
    It appears to group results by the number of terms in the counter, then apply
    conversion. Keeping for reference.

    Parameters:
    -----------
    processed_results : Dict[str, List[List[Any]]]
        Dictionary containing processed state entries [Counter, ...].
    methods : str, optional
        Method for conversion ('sympy' or 'string').

    Returns:
    --------
     Dict[str, List[Tuple[Union[str, Add, Mul, Ket, int], List[List[Any]]]]]
         Dictionary mapping original hash keys to lists of (expression, original_entry).
         (Structure seems potentially confusing).
    """
    print("Warning: `qs_generator` might be deprecated. Consider using `add_qc`.")
    save_quantum_expression: Dict[str, List[Tuple[Union[str, Add, Mul, Ket, int], List[List[Any]]]]] = defaultdict(list)

    for hash_key, entry_list in processed_results.items():
        for entry in entry_list:
            if entry and isinstance(entry[0], Counter):
                counter = entry[0]
                expression: Union[str, Add, Mul, Ket, int]
                if methods == 'sympy':
                    expression = counter_to_quantum_state(counter)
                elif methods == 'string':
                    terms = []
                    for state, coeff in sorted(counter.items()):
                         if coeff == 1:
                              terms.append(f'|{state}⟩')
                         elif coeff > 1:
                              terms.append(f'{coeff}|{state}⟩')
                    expression = ' + '.join(terms) if terms else '0'
                else:
                    raise ValueError("Method must be either 'string' or 'sympy'.")
                save_quantum_expression[hash_key].append((expression, entry))
            else:
                print(f"Warning: Invalid entry format for key {hash_key}. Skipping.")

    # The original code had grouping by len(counter), which seems odd here.
    # The logic below tries to mimic it but might not be the intended use.
    # It groups the generated (expression, entry) tuples based on the number of terms
    # in the original counter, which doesn't seem very useful.
    final_grouped_output: Dict[int, List[Tuple[Union[str, Add, Mul, Ket, int], List[List[Any]]]]] = defaultdict(list)
    for hash_key, expr_entry_list in save_quantum_expression.items():
        for expr, entry in expr_entry_list:
             if isinstance(entry[0], Counter):
                  num_terms = len(entry[0])
                  final_grouped_output[num_terms].append((expr, entry))

    # Returning the structure grouped by number of terms as per original code's apparent intent
    # return dict(final_grouped_output)
    # Returning the structure grouped by original hash_key seems more logical:
    return dict(save_quantum_expression)


def sorted_qs(
    quantum_expression_dict: Dict[Any, List[Tuple[Any, List[Any]]]]
    ) -> Dict[Any, List[Tuple[Any, List[Any]]]]:
    """
    Removes duplicate entries from lists within a dictionary based on the
    quantum expression and the original Counter content.

    DEPRECATED? Seems aimed at cleaning up the potentially confusing output of
    `qs_generator`. If `add_qc` is used, this might be unnecessary.

    Parameters:
    -----------
    quantum_expression_dict : Dict[Any, List[Tuple[Any, List[Any]]]]
        Dictionary where values are lists of (expression, original_entry) tuples.
        The original_entry is expected to be [Counter, ...].

    Returns:
    --------
    Dict[Any, List[Tuple[Any, List[Any]]]]
        Dictionary with duplicates removed from the value lists.
    """
    print("Warning: `sorted_qs` might be deprecated or need revision based on workflow.")
    save_sorted_qs: Dict[Any, List[Tuple[Any, List[Any]]]] = {}
    for key, expr_entry_list in quantum_expression_dict.items():
        unique_data: List[Tuple[Any, List[Any]]] = []
        seen: Set[Tuple[Any, Tuple[Tuple[str, int], ...]]] = set()

        for expr, original_entry in expr_entry_list:
            # Create a representation for uniqueness check
            # Includes the expression and a canonical tuple of the counter items
            if original_entry and isinstance(original_entry[0], Counter):
                counter = original_entry[0]
                counter_tuple = tuple(sorted(counter.items()))
                # Use str(expr) for SymPy expressions to make them hashable in the set
                seen_key = (str(expr), counter_tuple)

                if seen_key not in seen:
                    seen.add(seen_key)
                    unique_data.append((expr, original_entry))
            else:
                 print(f"Warning: Invalid original_entry format for key {key}. Skipping uniqueness check.")
                 # Decide whether to keep such entries or discard
                 # unique_data.append((expr, original_entry)) # Keep anyway

        save_sorted_qs[key] = unique_data
    return save_sorted_qs

