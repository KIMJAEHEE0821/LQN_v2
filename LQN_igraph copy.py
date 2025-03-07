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

def EPM_bipartite_graph_generator_igraph(num_system, num_ancilla):
    """
    igraph를 사용한 EPM 이분 그래프 생성기
    
    Parameters:
    -----------
    num_system : int
        시스템 노드 수
    num_ancilla : int
        앵커라 노드 수
    type : int
        그래프 유형
        
    Yields:
    -------
    igraph.Graph
        생성된 이분 그래프
    """
    import igraph as ig
    import itertools
    
    num_total = num_system + num_ancilla
    
    # red-blue 조합 생성 (type=0 가정)
    red_blue_combinations, num_combi = list_all_combinations_with_duplication(num_system, num_ancilla)
    
    # ancilla 조합 생성
    if num_ancilla > 0:
        ancilla_combinations_pre = generate_combinations(num_total)
        ancilla_combinations = list(itertools.product(ancilla_combinations_pre, repeat=num_ancilla))
    
    for rb_comb in red_blue_combinations:
        if num_ancilla > 0:
            for bl_comb in ancilla_combinations:
                # 노드 이름 생성
                system_node_names = [f'S_{i}' for i in range(num_system)]
                ancilla_node_names = [f'A_{i}' for i in range(num_ancilla)]
                sculpting_node_names = [str(i) for i in range(num_total)]
                
                # 모든 노드 이름 리스트
                all_node_names = system_node_names + ancilla_node_names + sculpting_node_names
                
                # igraph 그래프 생성
                G = ig.Graph()
                G.add_vertices(len(all_node_names))
                
                # 노드 이름 및 속성 설정
                G.vs["name"] = all_node_names
                
                # 노드 카테고리 설정
                categories = ["system_nodes"] * num_system + ["ancilla_nodes"] * num_ancilla + ["sculpting_nodes"] * num_total
                G.vs["category"] = categories
                
                # 이분 그래프 타입 설정 (0: 왼쪽 파티션, 1: 오른쪽 파티션)
                bipartite_types = [0] * (num_system + num_ancilla) + [1] * num_total
                G.vs["bipartite"] = bipartite_types
                
                # 엣지 생성을 위한 리스트
                edges = []
                edge_weights = []
                
                # red-blue 엣지 추가
                for rb_index, vt in enumerate(rb_comb):
                    red_idx = num_system + num_ancilla + vt[0]  # sculpting 노드 인덱스
                    blue_idx = num_system + num_ancilla + vt[1]  # sculpting 노드 인덱스
                    system_idx = rb_index  # system 노드 인덱스
                    
                    edges.append((system_idx, red_idx))
                    edge_weights.append(1.0)
                    
                    edges.append((system_idx, blue_idx))
                    edge_weights.append(2.0)
                
                # ancilla 엣지 추가
                for anc_index, vt_list in enumerate(bl_comb):
                    ancilla_idx = num_system + anc_index  # ancilla 노드 인덱스
                    for vt_inx in vt_list:
                        sculpting_idx = num_system + num_ancilla + vt_inx  # sculpting 노드 인덱스
                        edges.append((ancilla_idx, sculpting_idx))
                        edge_weights.append(3.0)
                
                # 그래프에 엣지 추가
                G.add_edges(edges)
                G.es["weight"] = edge_weights
                
                # 각 노드가 최소 2개 이상의 이웃을 갖는지 확인
                if all(G.degree(v) >= 2 for v in range(G.vcount())):
                    yield G
        else:  # num_ancilla == 0 인 경우
            # 노드 이름 생성
            system_node_names = [f'S_{i}' for i in range(num_system)]
            sculpting_node_names = [str(i) for i in range(num_total)]
            
            # 모든 노드 이름 리스트
            all_node_names = system_node_names + sculpting_node_names
            
            # igraph 그래프 생성
            G = ig.Graph()
            G.add_vertices(len(all_node_names))
            
            # 노드 이름 및 속성 설정
            G.vs["name"] = all_node_names
            
            # 노드 카테고리 설정
            categories = ["system_nodes"] * num_system + ["sculpting_nodes"] * num_total
            G.vs["category"] = categories
            
            # 이분 그래프 타입 설정
            bipartite_types = [0] * num_system + [1] * num_total
            G.vs["bipartite"] = bipartite_types
            
            # 엣지 생성을 위한 리스트
            edges = []
            edge_weights = []
            
            # red-blue 엣지 추가
            for rb_index, vt in enumerate(rb_comb):
                red_idx = num_system + vt[0]  # sculpting 노드 인덱스
                blue_idx = num_system + vt[1]  # sculpting 노드 인덱스
                system_idx = rb_index  # system 노드 인덱스
                
                edges.append((system_idx, red_idx))
                edge_weights.append(1.0)
                
                edges.append((system_idx, blue_idx))
                edge_weights.append(2.0)
            
            # 그래프에 엣지 추가
            G.add_edges(edges)
            G.es["weight"] = edge_weights
            
            # 각 노드가 최소 2개 이상의 이웃을 갖는지 확인
            if all(G.degree(v) >= 2 for v in range(G.vcount())):
                yield G

# igraph와 NetworkX 그래프 간 변환 유틸리티 함수
def igraph_to_networkx(g_igraph):
    """igraph 그래프를 NetworkX 그래프로 변환"""
    import networkx as nx
    
    G_nx = nx.Graph()
    
    # 노드 추가
    for v in g_igraph.vs:
        node_attrs = {attr: v[attr] for attr in v.attribute_names()}
        G_nx.add_node(v["name"], **node_attrs)
    
    # 엣지 추가
    for e in g_igraph.es:
        source = g_igraph.vs[e.source]["name"]
        target = g_igraph.vs[e.target]["name"]
        edge_attrs = {attr: e[attr] for attr in e.attribute_names()}
        G_nx.add_edge(source, target, **edge_attrs)
    
    return G_nx

def networkx_to_igraph(G_nx):
    """NetworkX 그래프를 igraph 그래프로 변환"""
    import igraph as ig
    
    g_igraph = ig.Graph()
    
    # 노드 추가
    node_names = list(G_nx.nodes())
    g_igraph.add_vertices(len(node_names))
    g_igraph.vs["name"] = node_names
    
    # 노드 속성 추가
    for attr in set().union(*(d.keys() for n, d in G_nx.nodes(data=True))):
        if attr != "name":  # 이름은 이미 설정됨
            g_igraph.vs[attr] = [G_nx.nodes[n].get(attr) for n in node_names]
    
    # 엣지 추가
    edges = [(node_names.index(u), node_names.index(v)) for u, v in G_nx.edges()]
    g_igraph.add_edges(edges)
    
    # 엣지 속성 추가
    for attr in set().union(*(d.keys() for u, v, d in G_nx.edges(data=True))):
        g_igraph.es[attr] = [G_nx.get_edge_data(u, v).get(attr) for u, v in G_nx.edges()]
    
    return g_igraph


# Canonical Form 생성 (가중치를 고려하지 않음)
def canonical_form_without_weights(ig_graph):
    # iGraph의 canonical_permutation을 사용하여 가중치 없이 처리
    perm = ig_graph.canonical_permutation()  # 색상(color) 정보 없이 permutation 생성
    permuted = ig_graph.permute_vertices(perm)  # 정렬 적용

    #병렬화를 위한 추가항
    perm = ig_graph.canonical_permutation()
    permuted = ig_graph.permute_vertices(perm)
    return tuple(map(tuple, permuted.get_adjacency().data))


# Canonical Form의 해시 생성
def generate_hash_from_canonical_form(canonical_form):
    # Canonical Form을 문자열로 변환한 뒤 해시값 생성
    canonical_str = str(canonical_form)
    return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()


# 그래프 리스트 처리 및 그룹화
def process_and_group_by_canonical_form(graph_list):
    canonical_groups = {}  # 해시 값을 키로, 그래프 그룹을 값으로 저장
    for graph in graph_list:
        # Canonical Form 생성 (가중치 고려 안 함)
        canonical_form = canonical_form_without_weights(graph)
        # Canonical Form의 해시 생성
        canonical_hash = generate_hash_from_canonical_form(canonical_form)
        # 동일 해시 값끼리 그룹화
        if canonical_hash not in canonical_groups:
            canonical_groups[canonical_hash] = []  # 새로운 그룹 생성
        canonical_groups[canonical_hash].append(graph)  # 그래프 추가
    return canonical_groups  # 그룹화된 결과 반환


def EPM_digraph_from_EPM_bipartite_graph_igraph(B):
    """
    igraph 버전의 EPM 이분 그래프(B)를 방향 그래프(D)로 변환
    
    Parameters:
    -----------
    B : igraph.Graph
        변환할 EPM 이분 그래프
        
    Returns:
    --------
    igraph.Graph
        방향성 있는 EPM 그래프
    """
    import igraph as ig
    import numpy as np
    
    # 시스템 및 앵커라 노드 식별
    system_nodes = [i for i, category in enumerate(B.vs["category"]) if category == "system_nodes"]
    ancilla_nodes = [i for i, category in enumerate(B.vs["category"]) if category == "ancilla_nodes"]
    
    num_system = len(system_nodes)
    num_ancilla = len(ancilla_nodes)
    num_total = num_system + num_ancilla
    
    # 노드 순서 준비 (시스템 노드, 앵커라 노드, 스컬프팅 노드 순)
    ordered_vertices = []
    ordered_vertices.extend(system_nodes)
    ordered_vertices.extend(ancilla_nodes)
    ordered_vertices.extend([i for i, category in enumerate(B.vs["category"]) if category == "sculpting_nodes"])
    
    # 인접 행렬 계산
    adj_matrix_B = np.array(B.get_adjacency(attribute="weight").data)
    
    # 재정렬된 인접 행렬 생성
    reordered_adj_matrix = np.zeros_like(adj_matrix_B)
    for i, v_i in enumerate(ordered_vertices):
        for j, v_j in enumerate(ordered_vertices):
            reordered_adj_matrix[i, j] = adj_matrix_B[v_i, v_j]
    
    # 방향 그래프용 관련 하위 행렬 추출
    adj_matrix_D = reordered_adj_matrix[:num_total, num_total:]
    
    # 방향 그래프 초기화
    D = ig.Graph(directed=True)
    D.add_vertices(num_total)
    
    # 노드 속성 설정
    categories = []
    node_names = []
    
    for i in range(num_total):
        if i < num_system:
            categories.append("system_nodes")
            node_names.append(f"S_{i}")
        else:
            categories.append("ancilla_nodes")
            node_names.append(f"A_{i-num_system}")
    
    D.vs["category"] = categories
    D.vs["name"] = node_names
    
    # 방향성 있는 엣지 추가
    edges = []
    weights = []
    
    for i in range(num_total):
        for j in range(num_total):
            if adj_matrix_D[i, j] != 0:
                # 방향은 j에서 i로 (원래 코드와 일치)
                edges.append((j, i))
                weights.append(adj_matrix_D[i, j])
    
    D.add_edges(edges)
    D.es["weight"] = weights
    
    return D

def is_single_scc_igraph(graph):
    """
    그래프가 단일 강연결 컴포넌트(SCC)인지 확인합니다.
    
    Parameters:
    -----------
    graph : igraph.Graph
        검사할 방향 그래프
        
    Returns:
    --------
    bool
        그래프가 단일 SCC이면 True, 아니면 False
    """
    # igraph에서는 strongly_connected_components() 메서드로 SCC를 찾습니다
    sccs = graph.connected_components(mode="strong")
    
    # 단일 SCC인지 확인: 컴포넌트가 1개이고 전체 노드를 포함해야 함
    return len(sccs) == 1 and len(sccs[0]) == graph.vcount()


def filter_groups_by_scc_igraph(grouped_graphs):
    """
    SCC 조건을 만족하는 그래프 그룹만 필터링합니다 (igraph 버전).
    그룹 구조(해시 키)를 유지합니다.
    
    Parameters:
    -----------
    grouped_graphs (dict): 
        해시 키와 igraph 그래프 리스트를 포함하는 딕셔너리
        
    Returns:
    --------
    dict: 
        SCC 조건을 만족하는 그래프 그룹만 포함하는 딕셔너리
    """
    filtered_groups = {}
    
    for key, graph_list in grouped_graphs.items():
        if len(graph_list) > 0:
            try:
                # 첫 번째 그래프로 SCC 확인
                first_graph = graph_list[0] 
                # 이분 그래프를 방향 그래프로 변환
                D = EPM_digraph_from_EPM_bipartite_graph_igraph(first_graph)
                
                if is_single_scc_igraph(D):
                    # SCC 조건을 만족하면 그룹 전체를 유지
                    filtered_groups[key] = graph_list
            except Exception as e:
                print(f"Error processing graph with key {key}: {e}")
                continue
    
    return filtered_groups

def extract_unique_bigraphs_with_weights_igraph(graph_list):
    """
    Extract unique bipartite graphs from a list, considering edge weights for isomorphism.
    
    Parameters:
    graph_list (list of ig.Graph): A list of bipartite graphs to process.
    
    Returns:
    list of ig.Graph: A list of unique bipartite graphs.
    """
    # List to store unique graphs
    unique_graphs = []
    
    for new_graph in graph_list:
        # Check if the new graph is isomorphic to any existing unique graph
        is_unique = True
        
        for existing_graph in unique_graphs:
            # 기본 검사: 노드 수와 엣지 수가 같은지 확인
            if new_graph.vcount() != existing_graph.vcount() or new_graph.ecount() != existing_graph.ecount():
                continue
                
            # 가중치 추출 (없으면 기본값 1 사용)
            new_weights = new_graph.es.get_attribute_values("weight") if "weight" in new_graph.edge_attributes() else [1] * new_graph.ecount()
            existing_weights = existing_graph.es.get_attribute_values("weight") if "weight" in existing_graph.edge_attributes() else [1] * existing_graph.ecount()
            
            # VF2 알고리즘으로 동형성 검사 - 가중치를 edge_color로 사용
            if new_graph.isomorphic_vf2(existing_graph, 
                                       edge_color1=new_weights,
                                       edge_color2=existing_weights):
                is_unique = False
                break
        
        # 고유한 그래프만 추가
        if is_unique:
            unique_graphs.append(new_graph)
    
    return unique_graphs

def extract_unique_bigraphs_from_groups_igraph(grouped_graphs):
    """
    그래프 그룹 딕셔너리에서 각 그룹 내의 고유한 그래프를 추출합니다.
    
    Parameters:
    -----------
    grouped_graphs (dict):
        해시 키를 키로, igraph 그래프 리스트를 값으로 가지는 딕셔너리
    
    Returns:
    --------
    dict:
        해시 키를 키로, 고유한 igraph 그래프 리스트를 값으로 가지는 딕셔너리
    """
    result = {}
    
    for key, graph_list in grouped_graphs.items():
        # 각 그래프가 igraph.Graph 인스턴스인지 확인
        valid_graphs = [g for g in graph_list if isinstance(g, ig.Graph)]
        
        if len(valid_graphs) != len(graph_list):
            print(f"경고: 키 {key}에 대해 {len(graph_list) - len(valid_graphs)}개의 유효하지 않은 그래프가 발견되었습니다.")
        
        # 각 그룹에 대해 고유한 그래프만 추출
        if valid_graphs:
            unique_graphs = extract_unique_bigraphs_with_weights_igraph(valid_graphs)
            result[key] = unique_graphs
        else:
            result[key] = []
    
    return result

def epm_process(num_system, num_ancilla):
    graph_generator = EPM_bipartite_graph_generator_igraph(num_system, num_ancilla)
    canonical_groups = process_and_group_by_canonical_form(graph_generator)
    filtered_groups = filter_groups_by_scc_igraph(canonical_groups)
    unique_bigraph = extract_unique_bigraphs_from_groups_igraph(filtered_groups)
    return unique_bigraph

##################################여기까지 새로운 함수 넣었음

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