import networkx as nx
import matplotlib.pyplot as plt
import random
from itertools import combinations, permutations
import itertools
import numpy as np
import pickle
from copy import deepcopy
import sympy as sp
from itertools import groupby
from collections import Counter, defaultdict
from functools import reduce
from math import gcd
from sympy.physics.quantum import Ket
from sympy import Add


import networkx as nx
import igraph as ig
import itertools
import hashlib

import itertools
import networkx as nx

# EPM bipartite graph 생성 함수
def EPM_bipartite_graph_generator_old(num_system, num_ancilla, type): 
    results = []  # 결과 저장 리스트
    num_total = num_system + num_ancilla

    # red-blue 조합 생성 (type=0 가정)
    red_blue_combinations, num_combi = list_all_combinations_with_duplication(num_system, num_ancilla)

    # ancilla 조합 생성
    ancilla_combinations_pre = generate_combinations(num_total)
    ancilla_combinations = list(itertools.product(ancilla_combinations_pre, repeat=num_ancilla))

    for rb_comb in red_blue_combinations:
        if num_ancilla != 0:
            for bl_comb in ancilla_combinations:
                B = nx.Graph()
                B.add_nodes_from(range(2 * num_total))
                mapping = {}

                # red-blue 엣지 추가
                for rb_index, vt in enumerate(rb_comb):
                    red = num_total + vt[0]
                    blue = num_total + vt[1]

                    B.add_edge(rb_index, red, weight=1.0)
                    B.add_edge(rb_index, blue, weight=2.0)

                    mapping[rb_index] = 'S_' + str(rb_index)
                    mapping[num_total + rb_index] = rb_index

                # ancilla 엣지 추가
                for anc_index, vt_list in enumerate(bl_comb):
                    for vt_inx in vt_list:
                        B.add_edge(num_system + anc_index, num_total + vt_inx, weight=3) 

                    mapping[num_system + anc_index] = 'A_' + str(anc_index)
                    mapping[num_total + num_system + anc_index] = num_system + anc_index

                # 노드 속성 추가
                for node_index in range(2 * num_total):
                    if node_index < num_system:
                        B.nodes[node_index]['category'] = 'system_nodes'
                        B.nodes[node_index]['bipartite'] = 0
                    elif node_index < num_total:
                        B.nodes[node_index]['category'] = 'ancilla_nodes'
                        B.nodes[node_index]['bipartite'] = 0
                    else:
                        B.nodes[node_index]['category'] = 'sculpting_nodes'
                        B.nodes[node_index]['bipartite'] = 1

                B = nx.relabel_nodes(B, mapping)  # 노드 이름 변경

                if all(len(list(B.neighbors(node))) >= 2 for node in B.nodes):
                    results.append(B)
        else:
            B = nx.Graph()
            B.add_nodes_from(range(2 * num_total))
            mapping = {}

            for rb_index, vt in enumerate(rb_comb):
                red = num_total + vt[0]
                blue = num_total + vt[1]

                B.add_edge(rb_index, red, weight=1.0)
                B.add_edge(rb_index, blue, weight=2.0)

                mapping[rb_index] = 'S_' + str(rb_index)
                mapping[num_total + rb_index] = rb_index

            for node_index in range(2 * num_total):
                if node_index < num_system:
                    B.nodes[node_index]['category'] = 'system_nodes'
                    B.nodes[node_index]['bipartite'] = 0
                elif node_index < num_total:
                    B.nodes[node_index]['category'] = 'ancilla_nodes'
                    B.nodes[node_index]['bipartite'] = 0
                else:
                    B.nodes[node_index]['category'] = 'sculpting_nodes'
                    B.nodes[node_index]['bipartite'] = 1

            B = nx.relabel_nodes(B, mapping)

            if all(len(list(B.neighbors(node))) >= 2 for node in B.nodes):
                results.append(B)

    return results  # 최종 결과 반환 


# NetworkX -> iGraph 변환
def nx_to_igraph(nx_graph):
    # 노드 매핑 (문자열 → 정수 변환)
    node_map = {node: idx for idx, node in enumerate(nx_graph.nodes)}

    # 엣지 추출
    edges = []
    for u, v in nx_graph.edges():
        edges.append((node_map[u], node_map[v]))  # 정수 인덱스로 변환된 엣지 추가

    # iGraph 객체 생성
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(node_map))  # 노드 추가
    ig_graph.add_edges(edges)  # 엣지 추가
    return ig_graph

# Canonical Form 생성 (가중치를 고려하지 않음)
def canonical_form_without_weights(ig_graph):
    # iGraph의 canonical_permutation을 사용하여 가중치 없이 처리
    perm = ig_graph.canonical_permutation()  # 색상(color) 정보 없이 permutation 생성
    permuted = ig_graph.permute_vertices(perm)  # 정렬 적용
    return tuple(map(tuple, permuted.get_adjacency().data))  # Immutable 변환

# Canonical Form의 해시 생성
def generate_hash_from_canonical_form(canonical_form):
    # Canonical Form을 문자열로 변환한 뒤 해시값 생성
    canonical_str = str(canonical_form)
    return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()

# 그래프 리스트 처리 및 그룹화
def process_and_group_by_canonical_form(graph_list):
    canonical_groups = {}  # 해시 값을 키로, 그래프 그룹을 값으로 저장
    for graph in graph_list:
        # NetworkX -> iGraph 변환
        ig_graph = nx_to_igraph(graph)
        # Canonical Form 생성 (가중치 고려 안 함)
        canonical_form = canonical_form_without_weights(ig_graph)
        # Canonical Form의 해시 생성
        canonical_hash = generate_hash_from_canonical_form(canonical_form)
        # 동일 해시 값끼리 그룹화
        if canonical_hash not in canonical_groups:
            canonical_groups[canonical_hash] = []  # 새로운 그룹 생성
        canonical_groups[canonical_hash].append(graph)  # 그래프 추가
    return canonical_groups  # 그룹화된 결과 반환

def get_adjacency_matrices(B):
    # Separate nodes by categories
    system_nodes = [node for node in B.nodes if B.nodes[node]['category'] == 'system_nodes']
    ancilla_nodes = [node for node in B.nodes if B.nodes[node]['category'] == 'ancilla_nodes']
    sculpting_nodes = [node for node in B.nodes if B.nodes[node]['category'] == 'sculpting_nodes']
    
    # Combine the nodes in the desired order
    sorted_nodes = system_nodes + ancilla_nodes + sculpting_nodes

    # Adjacency matrix (unweighted)
    #adjacency_matrix = nx.to_numpy_array(B, nodelist=sorted_nodes)

    # Weighted adjacency matrix
    weighted_adj_matrix = nx.to_numpy_array(B, nodelist=sorted_nodes, weight='weight')

    return weighted_adj_matrix\


def Draw_EPM_bipartite_graph(B):
    system_nodes = [node for node in list(B.nodes) if B.nodes[node]['category'] == 'system_nodes']
    ancilla_nodes = [node for node in list(B.nodes) if B.nodes[node]['category'] == 'ancilla_nodes']
    sculpting_nodes = [node for node in list(B.nodes) if B.nodes[node]['category'] == 'sculpting_nodes']
    num_system = len(system_nodes)
            
    pos = {}
    pos.update((node, (0, -index)) for index, node in enumerate(system_nodes))  # left nodes group 1
    pos.update((node, (1, -index)) for index, node in enumerate(sculpting_nodes[:num_system]))  # right nodes group 1 aligned with system_nodes
    pos.update((node, (0, -(index + len(system_nodes)))) for index, node in enumerate(ancilla_nodes))  # left nodes group 2
    pos.update((node, (1, -(index + len(system_nodes)))) for index, node in enumerate(sculpting_nodes[num_system:]))  # right nodes group 2 aligned with ancilla_nodes
    
    # Draw the undirected bipartite graph
    plt.figure(figsize=(6, 6))
    
    # Define node colors based on their category
    colors_B = ['lightblue' if B.nodes[node]['category'] == 'system_nodes' else 
                'lightcoral' if B.nodes[node]['category'] == 'ancilla_nodes' else 
                'lightgreen' for node in B.nodes]
    
    # Map the weights to specific colors for visualization
    edge_colors_B = ['red' if B[u][v]['weight'] == 1 else 
                     'blue' if B[u][v]['weight'] == 2 else 
                     'black' for u, v in B.edges]
    
    nx.draw(B, pos, with_labels=True, node_color=colors_B, edge_color=edge_colors_B, width=2)
    plt.title("Undirected Bipartite Graph")
    plt.show()


import numpy as np
import networkx as nx

def EPM_digraph_from_EPM_bipartite_graph(B):
    """
    Convert an EPM bipartite graph (B) to a directed graph (D).

    Parameters:
        B (nx.Graph): Input bipartite graph.

    Returns:
        nx.DiGraph: Directed graph derived from the input bipartite graph.
    """
    # Initialize directed graph
    D = nx.DiGraph()

    # Identify system and ancilla nodes
    system_nodes = [node for node in B.nodes if B.nodes[node]['category'] == 'system_nodes']
    ancilla_nodes = [node for node in B.nodes if B.nodes[node]['category'] == 'ancilla_nodes']
    num_system = len(system_nodes)
    num_ancilla = len(ancilla_nodes)
    num_total = num_system + num_ancilla

    # Generate adjacency matrix and weight matrix from B
    adj_weight_matrix_B = get_adjacency_matrices(B)

    # Extract relevant submatrices for the directed graph
    # adj_matrix_D = adj_matrix_B[:num_total, num_total:]
    adj_weight_matrix_D = adj_weight_matrix_B[:num_total, num_total:]

    # Add nodes to directed graph
    D.add_nodes_from(range(num_total))

    # Add directed edges with weights
    for i in range(num_total):
        for j in range(num_total):
            if adj_weight_matrix_D[i, j] != 0:
                D.add_edge(j, i, weight=adj_weight_matrix_D[i, j])  # Reverse direction

    # Map node labels and categories
    mapping = {}
    for i in range(num_total):
        if i < num_system:
            mapping[i] = 'S_' + str(i)
            D.nodes[i]['category'] = 'system_nodes'
        else:
            mapping[i] = 'A_' + str(i - num_system)
            D.nodes[i]['category'] = 'ancilla_nodes'

    # Relabel nodes
    D = nx.relabel_nodes(D, mapping)

    return D

def Draw_EPM_digraph(D):
    """
    Visualize an EPM directed graph (D).

    Parameters:
        D (nx.DiGraph): Directed graph to visualize.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    # Identify system and ancilla nodes
    system_nodes = [node for node in D.nodes if D.nodes[node]['category'] == 'system_nodes']
    ancilla_nodes = [node for node in D.nodes if D.nodes[node]['category'] == 'ancilla_nodes']
    
    # Define positions for nodes
    pos_D = {}
    pos_D.update((node, (0, index)) for index, node in enumerate(system_nodes))  # System nodes on the left
    pos_D.update((node, (1, index + len(system_nodes))) for index, node in enumerate(ancilla_nodes))  # Ancilla nodes on the right
    
    # Set the figure size
    plt.figure(figsize=(12, 6))
    
    # Define node colors based on their category
    colors_D = ['lightblue' if D.nodes[node]['category'] == 'system_nodes' else 'lightcoral' for node in D.nodes]
    
    # Map the edge weights to specific colors for visualization
    edge_colors_D = ['red' if D[u][v]['weight'] == 1 else 
                     'blue' if D[u][v]['weight'] == 2 else 
                     'black' for u, v in D.edges]  # Use red for weight 1, blue for weight 2, black otherwise
    
    # Define curved edges
    curved_edges = [edge for edge in D.edges]
    arc_rad = 0.25  # Radius for edge curvature
    
    # Draw the graph
    nx.draw(D, pos_D, with_labels=True, node_color=colors_D, edge_color=edge_colors_D, 
            width=2, arrows=True, connectionstyle=f'arc3,rad={arc_rad}')
    plt.title("Directed Graph with Curved Edges")
    plt.show()


def is_single_scc(graph):
    """
    Check if the graph is a single strongly connected component (SCC).

    Parameters:
        graph (nx.DiGraph): Directed graph.

    Returns:
        bool: True if the graph is a single SCC, False otherwise.
    """
    sccs = list(nx.strongly_connected_components(graph))
    return len(sccs) == 1 and len(sccs[0]) == len(graph)

def filter_groups_by_scc(grouped_graphs):
    """
    Filter groups of graphs based on whether their first element's DiGraph is a single SCC.

    Parameters:
        grouped_graphs (dict): Dictionary where each key represents a group identifier, and the value is a list of graphs.

    Returns:
        dict: Filtered dictionary containing only groups whose first graph is a single SCC.
    """
    filtered_groups = {}
    for key, group in grouped_graphs.items():
        if group:
            # Convert the first graph in the group to a DiGraph
            D = EPM_digraph_from_EPM_bipartite_graph(group[0])
            # Check if it is a single SCC
            if is_single_scc(D):
                filtered_groups[key] = group
    return filtered_groups


#아래가 원본임. 병렬처리한 부분으로 바꿔보는중. 
def list_all_combinations_with_duplication(num_system, num_ancilla):
    # Generate all possible ordered pairs (i, j) where i != j
    p = num_system+num_ancilla
    vertices = list(range(p))
    all_pairs = list(itertools.permutations(vertices, 2)) # 중복 허용하고, 시스템 vertex를 반대편 vertex에 연결할 모든 경우의 수를 따짐. 시스템 vertex는 반대편에 2개의 edge를 연결할 수 있음. 
    
    # Generate all combinations of these pairs taken n at a time with repetition allowed
    all_combinations = list(itertools.product(all_pairs, repeat=num_system)) 
    
    return all_combinations, len(all_combinations)


def generate_combinations(n):
    all_combinations = []
    elements = list(range(n))
    for i in range(1,n+1):
        combinations = list(itertools.combinations(elements, i))
        all_combinations.extend(combinations)
    return all_combinations


def EPM_di_graph_generator(result_of_the_bipartite_list):
    save_result_digraph = []
    for i in range(len(result_of_the_bipartite_list)):
        save_result_digraph.append(EPM_digraph_from_EPM_bipartite_graph(
            result_of_the_bipartite_list[i][0],
            adj_matrix_B=result_of_the_bipartite_list[i][1],
            adj_weight_matrix_B=result_of_the_bipartite_list[i][2]
        ))
    return save_result_digraph

def graph_generation(num_system, num_ancilla, type):
    bigraph_list_2 = []
    digraph_list_2 = []

    unique_bigraphs = set()  # Store unique bigraphs for isomorphism check

    i = 0

    for bigraph in EPM_bipartite_graph_generator(num_system, num_ancilla):
        # Generate digraph from bipartite graph
        #digraph = EPM_digraph_from_EPM_bipartite_graph(bigraph)
        i += 1

        # Define edge_match to consider edge weights
        edge_match = lambda x, y: x.get("weight", 1) == y.get("weight", 1)

        # Check uniqueness of the bipartite graph
        is_unique = True
        for existing_bigraph in unique_bigraphs:
            if nx.is_isomorphic(bigraph, existing_bigraph, edge_match=edge_match):
                is_unique = False
                break

        if is_unique:
            unique_bigraphs.add(bigraph)  # Add unique bigraph
            bigraph_list_2.append(bigraph) 
            digraph = EPM_digraph_from_EPM_bipartite_graph(bigraph)
            digraph_list_2.append(digraph)

    return bigraph_list_2, digraph_list_2


def count_dict_elements(d):
    """
    Count the total number of elements in a dictionary, including keys and values.

    Parameters:
        d (dict): Input dictionary.

    Returns:
        int: Total number of elements in the dictionary.
    """
    if not isinstance(d, dict):
        raise ValueError("Input must be a dictionary.")
    
    total_elements = 0
    
    for key, value in d.items():
        total_elements += 1  # Count the key
        if isinstance(value, (list, dict)):  # Recursively count elements in lists or dictionaries
            total_elements += count_dict_elements(value) if isinstance(value, dict) else len(value)
        else:
            total_elements += 1  # Count the value if not a container
    
    return total_elements

def extract_unique_bigraphs_with_weights(graph_list):
    """
    Extract unique bipartite graphs from a list, considering edge weights for isomorphism.

    Parameters:
        graph_list (list of nx.Graph): A list of bipartite graphs to process.

    Returns:
        list of nx.Graph: A list of unique bipartite graphs.
    """
    # List to store unique graphs
    unique_graphs = []

    # Define edge_match to compare edge weights
    edge_match = lambda x, y: x.get("weight", 1) == y.get("weight", 1)
    
    for new_graph in graph_list:
        # Check if the new graph is isomorphic to any existing unique graph
        is_unique = True
        for existing_graph in unique_graphs:
            if nx.is_isomorphic(new_graph, existing_graph, edge_match=edge_match):
                is_unique = False
                break

        # If unique, add to the unique_graphs list
        if is_unique:
            unique_graphs.append(new_graph)

    return unique_graphs

##################################여기까지 새로운 함수 넣었음

def is_perfect_matching(G, matching):
    # 완전 매칭은 모든 노드를 커버합니다 (이분 그래프의 경우 전체 노드의 절반)
    return len(matching) == len(G.nodes) // 2

def dfs_all_matchings(G, U, V, current_matching, all_matchings, all_weights, matched, u_index=0):
    if is_perfect_matching(G, current_matching):#U는 왼쪽 vertex, V는 오른쪽 vertex
        #print(current_matching)
        all_matchings.append(current_matching[:])
        
        # Construct the weight string at the same time, excluding '+'여기서 +는 |+\rangle가 아님 문자 +
        weights = ''.join([G[u][v]['weight'] for u, v in current_matching if G[u][v]['weight'] != '+'])
        all_weights.append(weights)
        
        return

    # 현재 인덱스부터 시작하여 U를 반복하여 불필요한 확인을 피합니다
    for i in range(u_index, len(U)): #U vertex list\
        u = U[i]
        #print(u, i)
        if matched[u]: # matched[u]는 해당 vertex가 perfect matching 찾을때 쓰였는지 여부 조사
            continue #matched[u]가 있으면 다음 for loop로 넘어감

        for v in V:
            #print(u,v, 'before')
            if (u, v) in G.edges and not matched[v]:
                #print(u,v, 'after')
                current_matching.append((u, v))
                #print(current_matching)
                matched[u] = True
                matched[v] = True
                #print(u,v, matched[u], matched[v])

                # 다음 인덱스로 U에 대해 재귀 호출
                #print(i, 'check')
                dfs_all_matchings(G, U, V, current_matching, all_matchings, all_weights, matched, i + 1)
                #print(i, 'check2')

                # 백트래킹
                matched[u] = False
                matched[v] = False
                #print(u,v, matched[u], matched[v], 'back')
                #print(current_matching, 'back')
                current_matching.pop()
                #print(current_matching, 'pop')

def find_all_perfect_matchings(G):
    U = [n for n in G.nodes if G.nodes[n]['bipartite'] == 0] #bipartite 그래프는 두 그룹으로 나뉘어짐 왼쪽 vertex와 오른쪽 vertex 그 중에서 왼쪽 vertex를 0 그룹으로 둔거
    V = [n for n in G.nodes if G.nodes[n]['bipartite'] == 1]
    all_matchings = []
    all_weights = []  # List to store the weight strings
    matched = {node: False for node in G.nodes}

    dfs_all_matchings(G, U, V, [], all_matchings, all_weights, matched) ##여기 나오는 []는 dfs_all_matghings에서 current matching에 해당함. 
    
    return all_matchings, all_weights

def find_unused_edges(G, perfect_matchings):
    # Create a set of all edges used in any perfect matching
    used_edges = []
    for matching in perfect_matchings:
        for edge in matching:
            used_edges.append((edge[0], edge[1]))  # Only consider the nodes, not the weight

    # Find all unused edges by comparing with the edges in the graph
    unused_edges = [edge for edge in G.edges if edge not in used_edges]
    return unused_edges

def find_pm_of_bigraph(bigraph_result, digraph_result):
    save_fw_results = []

    for i in range(len(bigraph_result)):
        perfect_matching_result, weight_strings = find_all_perfect_matchings(bigraph_result[i][0])
        if not find_unused_edges(bigraph_result[i][0], perfect_matching_result) and len(perfect_matching_result) >= 2:
            save_fw_results.append([Counter(weight_strings), perfect_matching_result, bigraph_result[i], digraph_result[i]])
    for i in save_fw_results:
        values = list(i[0].values()) #결과에서 state들의 coefficient를 최대공약수로 나눔. 
        gcd_of_values = reduce(gcd, values)
        for key in i[0]:
            i[0][key] //= gcd_of_values
    return save_fw_results

def remove_same_state(save_fw_results):
    # Set to keep track of unique dictionaries
    seen = set()

    # List to store only the first occurrence of each unique dictionary
    unique_counters = []

    # Iterate over each Counter object
    for counter in save_fw_results:
        # Convert Counter to a tuple of sorted items for comparison
        counter_tuple = tuple(sorted(counter[0].items()))
        
        # If this tuple hasn't been seen before, add it to the list and the set
        if counter_tuple not in seen:
            seen.add(counter_tuple)
            unique_counters.append(counter)
    return unique_counters

def gen_grouped_counters(unique_counters):
    # Group dictionaries by their value frequencies
    grouped_counters = defaultdict(list)

    for counter in unique_counters:
        # Convert the values of the Counter to a sorted tuple to use as a key
        value_tuple = tuple(sorted(counter[0].values()))
        grouped_counters[value_tuple].append(counter)
    return grouped_counters


# 키의 자릿수를 바꾸는 함수 (예: 1번째, 2번째, 3번째 자리 순서를 바꿈)
def apply_permutation_to_keys(counter, permutation):
    transformed_counter = Counter()
    for key, value in counter.items():
        new_key = ''.join(key[i] for i in permutation)
        transformed_counter[new_key] = value
    return transformed_counter

# 가능한 키 순열들(자릿수 변환)을 생성한 후 Counter를 비교하여 중복을 제거하는 함수
def remove_duplicate_counters(data):
    indices_to_remove = set()  # 중복된 Counter들의 인덱스를 저장
    unique_counters = []  # 고유한 Counter들을 저장할 리스트
    n_elements = len(data)

    for i in range(n_elements):
        if i in indices_to_remove:
            continue  # 이미 제거된 Counter는 건너뜀

        counter1 = data[i][0]  # 첫 번째 요소는 Counter
        keys1 = list(counter1.keys())
        n_bits = len(keys1[0])  # 첫 번째 키의 길이
        all_permutations = permutations(range(n_bits))  # 키 길이에 따른 모든 순열 생성
        
        # 중복을 발견했을 때는 하나만 남겨야 하므로 unique_counters에 저장
        unique_counters.append(data[i])

        for permutation in all_permutations:
            # counter1에 순열 적용
            transformed_counter1 = apply_permutation_to_keys(counter1, permutation)

            for j in range(i + 1, n_elements):
                if j in indices_to_remove:
                    continue  # 이미 제거된 Counter는 건너뜀
                
                counter2 = data[j][0]
                if transformed_counter1 == counter2:
                    indices_to_remove.add(j)  # 중복된 Counter인 경우 인덱스를 기록

    # 중복된 요소들을 제거하고 고유한 Counter만 반환
    return [data[k] for k in range(n_elements) if k not in indices_to_remove]

def remove_duplicate_counters_full_list(grouped_counters):
    save_filtered_data = {}
    for i in range(len(list(grouped_counters.keys()))):
        save_filtered_data[list(grouped_counters.keys())[i]]= remove_duplicate_counters(grouped_counters[list(grouped_counters.keys())[i]])
    return save_filtered_data

# Function to flip a specific bit in a binary string
def flip_bit_string(binary_str, n):
    bit_list = list(binary_str)
    bit_list[n] = '1' if bit_list[n] == '0' else '0'
    return ''.join(bit_list) # list를 다시 bitstring으로 바꾸는거

# Function to flip multiple bits based on given positions
def flip_multiple_bits(binary_str, positions):
    for pos in positions:
        binary_str = flip_bit_string(binary_str, pos)
    return binary_str

# Function to check if two Counters can be made identical by flipping bits
def find_transformation(counter1, counter2):
    bit_length = len(next(iter(counter1))) #그냥 counter에 있는 bit string길이 나타남
    for r in range(1, bit_length + 1):
        for positions in combinations(range(bit_length), r): #이러면 (0,1,2,...bit_length-1)에서 r개 뽑는 경우의 수 다 뽑아냄. 즉 bit flip 어디할지 보여주는거.
            flipped_data = Counter([flip_multiple_bits(bstr, positions) for bstr in counter1])
            if flipped_data == counter2:
                return positions  # Return the positions of the bits that need to be flipped
    return None  # Return None if no transformation is found

# Optimized function to check all pairs of Counters and avoid redundant checks
def check_transformations(data_list):
    transformations = []
    transformable_indices = set()
    
    # Extract the Counter objects from the complex data structure
    counters = [item[0] for item in data_list] #데이터에서 Counter부분만 뽑아냄
    
    for i, counter1 in enumerate(counters):
        if i in transformable_indices:
            continue  # Skip if this Counter is already known to be transformable
        
        for j, counter2 in enumerate(counters):
            if i != j and j not in transformable_indices:
                bit_positions = find_transformation(counter1, counter2)
                if bit_positions is not None:
                    transformations.append((i, j, bit_positions))
                    #transformable_indices.add(i)
                    transformable_indices.add(j)
                    
    non_transformable = [data_list[i] for i in range(len(counters)) if i not in transformable_indices]
    #non_transformable = [counters[i] for i in range(len(counters)) if i not in transformable_indices]
    
    return transformations, non_transformable

def check_tranf_full_data(save_filtered_data):
    test_remain_state = {}
    for i in save_filtered_data.keys():
        transformations, non_transformable_data = check_transformations(save_filtered_data[i])
        test_remain_state[i] = non_transformable_data
    return test_remain_state

# 단일 큐비트 Hadamard 게이트와 단위 행렬 정의
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
I = np.eye(2)

# 특정 큐비트 하위 집합에 Hadamard 게이트를 적용하는 함수
def apply_hadamard_to_qubits(n_qubits, hadamard_qubits):
    operator = 1
    for i in range(n_qubits):
        if i in hadamard_qubits:
            operator = np.kron(operator, H)
        else:
            operator = np.kron(operator, I)
    return operator

# Counter로부터 양자 상태 벡터를 정규화하고 생성하는 함수
def counter_to_state_vector(counter, n_qubits):
    total_counts = sum(counter.values())
    state_vector = np.zeros(2**n_qubits, dtype=complex)
    
    for bitstring, count in counter.items():
        index = int(bitstring, 2)  # 비트 문자열을 정수 인덱스로 변환
        amplitude = np.sqrt(count / total_counts)  # 진폭을 정규화
        state_vector[index] = amplitude
    return state_vector


def get_unique_transformed_states_from_dict_v2(data):
    all_transformed_states = []  # 모든 상태를 저장할 리스트
    original_states = []
    unique_states_dict = {}
    transformable_indices = set()
    
    for key, counters in data.items():
        unique_transformed_states = []
        #print(key)
        
        for order, counter in enumerate(counters): 
            state_vector = counter_to_state_vector(counter[0], len(counter[0].most_common()[0][0]))
           
            
            # 변환된 상태들을 저장할 리스트
            transformed_states = []
            
            # 큐비트 하위 집합마다 Hadamard 게이트 적용
            for k in range(1, len(counter[0].most_common()[0][0]) + 1):  # 그냥 qubit 수임
                for qubits in combinations(range(len(counter[0].most_common()[0][0])), k):
                    H_subset = apply_hadamard_to_qubits(len(counter[0].most_common()[0][0]), qubits)
                    final_state = np.dot(H_subset, state_vector)
                    transformed_states.append(final_state)
            
            # 모든 상태를 전역 리스트에 추가
            all_transformed_states.append((key, order, transformed_states))  # 원본 상태와 변환된 상태를 함께 저장
            original_states.append((key, order, state_vector))
            
        unique_states_dict[key] = unique_transformed_states  # 임시 저장 (나중에 갱신)

    # 이제 모든 key에 대해 중복 상태를 확인
    final_unique_states = []
    for (key1, order1, transformed_states_i) in all_transformed_states:
        if (key1, order1) in transformable_indices:#여기가 문제임(order는 각 key별로 순서를 매긴 거여서, 이거만 따지면 안됨.)
            continue
        is_unique = True  # 현재 상태가 고유한지 여부를 추적

        # 다른 상태들과 비교하여 중복 여부 확인
        for (key2, order2, state_vector_j) in original_states:
            if (key1, order1)!=(key2, order2) and (key2, order2) not in transformable_indices: ## 이거 맞는지 확인
                # 서로 다른 상태들 간에 비교
                for t_state_i in transformed_states_i:
                    if np.allclose(t_state_i, state_vector_j):
                        is_unique = False  # 중복 상태 발견
                        transformable_indices.add(key2, order2)
                        final_unique_states.append((key1, order1))
                        break
                if not is_unique:
                    break
            if not is_unique:
                break

        # 고유한 상태인 경우만 저장
        if is_unique:
            final_unique_states.append((key1, order1))  # 중복이 없는 원본 상태만 저장

    # unique_states_dict를 갱신
    for key in unique_states_dict:
        for key_2, b in final_unique_states:
            if key == key_2:
                unique_states_dict[key].append(data[key][b])

    return unique_states_dict

# Define the load_results function
def load_results(num_system, num_ancilla, type):
    # Generate file names based on the input parameters
    bigraph_filename = f'bigraph_result_sys{num_system}_anc{num_ancilla}_type{type}.pkl'
    digraph_filename = f'digraph_result_sys{num_system}_anc{num_ancilla}_type{type}.pkl'

    # Load results from files
    with open(bigraph_filename, 'rb') as f:
        bigraph_result = pickle.load(f)

    with open(digraph_filename, 'rb') as f:
        digraph_result = pickle.load(f)

    print(f"Results loaded from '{bigraph_filename}' and '{digraph_filename}'")
    return bigraph_result, digraph_result

def convert_to_quantum_notation(formatted_weight_lists, method='string'):
    """
    Convert a list of lists of binary strings to quantum state notation.
    
    Parameters:
    formatted_weight_lists (list of lists): List containing lists of binary strings representing quantum states.
    method (str): Method to use for conversion ('string' or 'sympy').
    
    Returns:
    list: A list of quantum state expressions, each element being a string or a sympy expression.
    """
    
    all_expressions = []

    for formatted_weights in formatted_weight_lists:
        if method == 'string':
            # String-based method
            quantum_states = [f'|{state}⟩' for state in formatted_weights]
            quantum_expression = ' + '.join(quantum_states)
        elif method == 'sympy':
            # SymPy-based method
            quantum_states = [sp.symbols(f'|{state}⟩') for state in formatted_weights]
            quantum_expression = sp.Add(*quantum_states)
        else:
            raise ValueError("Method must be either 'string' or 'sympy'.")
        
        # Wrap each expression in a list to match the desired output format
        all_expressions.append([quantum_expression, formatted_weights])
    
    return all_expressions

def qs_generator(formatted_weight_results, methods):
    sorted_formatted = sorted(formatted_weight_results, key=len)
    grouped_formatted = {k: list(v) for k, v in groupby(sorted_formatted, key=len)}
    save_quantum_expression = {}
    for i, j in grouped_formatted.items():
        save_quantum_expression[i] = convert_to_quantum_notation(j, method = methods)
    return save_quantum_expression
####################################아래 함수는 LQN_utils_state.py에 없다

def sorted_qs(quantum_expression):
    save_sorted_qs = {}
    for i in quantum_expression.keys():
        # Remove duplicates
        unique_data = []
        seen = set()
        for sym_expr, string_list in quantum_expression[i]:
            # Create a tuple for comparison
            tuple_repr = (sym_expr, tuple(string_list))
            
            # Add to the list if not seen
            if tuple_repr not in seen:
                seen.add(tuple_repr)
                unique_data.append([sym_expr, string_list])
        save_sorted_qs[i] = unique_data
    return save_sorted_qs

def counter_to_quantum_state(counter):
    # List to store the quantum states
    quantum_state_terms = []
    
    # Iterate over the Counter's keys (states) and values (coefficients)
    for state, coeff in counter.items():
        # Create a ket for each state and multiply by its coefficient
        quantum_state_terms.append(coeff * Ket(state))
    
    # Sum all the terms to get the final quantum state
    return Add(*quantum_state_terms)

def add_qc(unique_states_result):
    for value_list in unique_states_result.values():
        for value in value_list:
            if isinstance(value[0], Counter):
                quantum_state = counter_to_quantum_state(value[0])
                value.append(quantum_state)
    return unique_states_result