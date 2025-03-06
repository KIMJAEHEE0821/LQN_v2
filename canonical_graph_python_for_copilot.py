import networkx as nx
import matplotlib.pyplot as plt
import random
from LQN_utils_state_save_parallel import *
import itertools
from copy import deepcopy 
import sympy as sp
from itertools import groupby
import hashlib
import networkx as nx
import igraph as ig
import itertools
import hashlib

import networkx as nx
import hashlib
import itertools
from collections import defaultdict

import networkx as nx
import itertools
from collections import defaultdict
import hashlib

import networkx as nx
import itertools

def EPM_bipartite_graph_generator(num_system, num_ancilla, type): 
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

                # ✅ 노드 속성 추가 (relabel_nodes() 실행 전)
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

                # ✅ 속성이 보존되는지 확인하면서 노드 이름 변경
                B = nx.relabel_nodes(B, mapping, copy=False)  # ✅ copy=False로 속성 유지
                
                # relabel 이후 속성을 재적용 (보존되지 않은 경우 대비)
                for node, new_label in mapping.items():
                    if new_label in B.nodes:  # 새로운 노드에 속성 적용
                        B.nodes[new_label]['category'] = B.nodes[node].get('category', 'unknown')
                        B.nodes[new_label]['bipartite'] = B.nodes[node].get('bipartite', -1)

                if all(len(list(B.neighbors(node))) >= 2 for node in B.nodes):
                    yield B  # ✅ 메모리에 저장하지 않고 바로 반환
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

            # ✅ 노드 속성 추가 (relabel_nodes() 실행 전)
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

            B = nx.relabel_nodes(B, mapping, copy=False)  # ✅ copy=False로 속성 유지

            # ✅ relabel 이후 속성을 재적용
            for node, new_label in mapping.items():
                if new_label in B.nodes:
                    B.nodes[new_label]['category'] = B.nodes[node].get('category', 'unknown')
                    B.nodes[new_label]['bipartite'] = B.nodes[node].get('bipartite', -1)

            if all(len(list(B.neighbors(node))) >= 2 for node in B.nodes):
                yield B  # ✅ 제너레이터 사용

def nx_to_igraph(nx_graph):
    node_map = {node: idx for idx, node in enumerate(nx_graph.nodes)}
    edges = [(node_map[u], node_map[v]) for u, v in nx_graph.edges()]
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(node_map))
    ig_graph.add_edges(edges)
    return ig_graph

def canonical_form_without_weights(ig_graph):
    perm = ig_graph.canonical_permutation()
    permuted = ig_graph.permute_vertices(perm)
    return tuple(map(tuple, permuted.get_adjacency().data))

def generate_hash_from_canonical_form(canonical_form):
    canonical_str = str(canonical_form)
    return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()

def process_and_group_by_canonical_form(graph_iter):
    canonical_groups = defaultdict(list)
    for graph in graph_iter:
        ig_graph = nx_to_igraph(graph)
        canonical_form = canonical_form_without_weights(ig_graph)
        canonical_hash = generate_hash_from_canonical_form(canonical_form)

        if canonical_hash not in canonical_groups:
            def graph_generator():
                yield graph  # ✅ 첫 번째 그래프만 메모리에 유지
            canonical_groups[canonical_hash] = graph_generator()

    return canonical_groups  # ✅ 제너레이터 유지 (메모리 절약)

def is_single_scc(graph):
    sccs = list(nx.strongly_connected_components(graph))
    return len(sccs) == 1 and len(sccs[0]) == len(graph)

def filter_groups_by_scc(grouped_graphs):
    """
    Filter groups of graphs based on whether their first element's DiGraph is a single SCC.

    Parameters:
        grouped_graphs (dict): Dictionary where each key represents a group identifier, 
                               and the value is a generator of graphs.

    Returns:
        dict: Filtered dictionary containing only groups whose first graph is a single SCC.
    """
    filtered_groups = {}

    for key, graph_gen in grouped_graphs.items():
        try:
            first_graph = next(graph_gen)  # ✅ 제너레이터에서 첫 번째 그래프 가져오기
            D = EPM_digraph_from_EPM_bipartite_graph(first_graph)  # ✅ 이제 정상적으로 처리 가능!
            if is_single_scc(D):
                filtered_groups[key] = graph_gen  # ✅ 제너레이터 유지
        except StopIteration:
            continue

    return filtered_groups

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

    # ✅ 정수와 문자열을 분리하여 정렬한 후 결합
    sorted_nodes = sorted([node for node in B.nodes if isinstance(node, int)]) + \
                   sorted([node for node in B.nodes if isinstance(node, str)])

    # ✅ get_adjacency_matrices()에 정렬된 노드 리스트 전달
    adj_weight_matrix_B = get_adjacency_matrices(B, nodelist=sorted_nodes)

    # Extract relevant submatrices for the directed graph
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

def extract_unique_bigraphs_with_weights(filtered_groups):
    """
    Extract unique bipartite graphs from filtered_groups while considering edge weights.

    Parameters:
        filtered_groups (dict): A dictionary where values are graph generators.

    Returns:
        dict: A dictionary where keys are canonical hashes, and values are lists of unique graphs.
    """
    unique_graphs_list = {}
    edge_match = lambda x, y: x.get("weight", 1) == y.get("weight", 1)

    for key, graph_gen in filtered_groups.items():
        graph_list = list(graph_gen)  # ✅ 제너레이터를 리스트로 변환
        unique_graphs = []

        for new_graph in graph_list:
            is_unique = True
            for existing_graph in unique_graphs:
                if nx.is_isomorphic(new_graph, existing_graph, edge_match=edge_match):
                    is_unique = False
                    break

            if is_unique:
                unique_graphs.append(new_graph)

        unique_graphs_list[key] = unique_graphs  
        return unique_graphs_list  

# 1️⃣ 제너레이터 기반으로 그래프 생성
graph_generator = EPM_bipartite_graph_generator(num_system=3, num_ancilla=1, type=0)