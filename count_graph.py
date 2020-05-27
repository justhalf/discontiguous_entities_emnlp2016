# -*- coding: utf-8 -*-
"""
A script to calculate the number of subgraph in our model
"""

# Import statements
from __future__ import division, print_function
from numpy.linalg import matrix_power
import numpy as np
import sys
from itertools import product
from pprint import pprint
import argparse

def to_bin(num, length):
    result = [0]*length
    idx = -1
    while num > 0:
        result[idx] = num%2
        num //= 2
        idx -= 1
    return tuple(result)

def print_matrix(mat):
    column_width = [max(map(lambda x: len(str(x)), col)) for col in mat.transpose().tolist()]
    for row in mat.tolist():
        for idx, num in enumerate(row):
            if idx > 0:
                print(' ', end='')
            print('{value:{length}d}'.format(value=num, length=column_width[idx]), end='')
        print()

def test_simple():
    # Simple grid
    init = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=int).transpose()
    mul_mat = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0, 0, 0, 0],
                          [0, 1, 0, 2, 0, 0, 0, 0],
                          [0, 0, 1, 0, 1, 0, 1, 0],
                          [0, 0, 0, 1, 0, 1, 0, 1],
                          [0, 0, 2, 1, 0, 1, 2, 3],
                          [0, 0, 0, 3, 0, 1, 0, 5]], dtype=int)
    init2 = np.array([1, 0, 0, 0], dtype=int).transpose()
    mul_mat2 = np.array([[1, 0, 0, 0],
                         [1, 2, 0, 0],
                         [0, 1, 1, 1],
                         [0, 3, 1, 5]], dtype=int)
    print('f(1)')
    print(init.transpose())
    print(mul_mat)
    vec = init
    n = 5
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    # Direct
    mul_mat = matrix_power(mul_mat, n-1)
    print(mul_mat)
    print((mul_mat.dot(vec)).transpose())

    mul_mat2 = matrix_power(mul_mat2, n-1)
    print(mul_mat2)
    print((mul_mat2.dot(init2)).transpose())

    # Iterative
    # for i in range(2, n+1):
    #     vec = mul_mat.dot(vec)
    #     print('f({})'.format(i))
    #     print(vec.transpose())

def get_next_states(idx, end_anywhere, all_components):
    if idx % 2 == 1:  # O-nodes
        return [2**idx, 2**(idx-1), 2**(idx)|2**(idx-1)]  # [next-O-node, next-B-node, both]
    if idx == 0:
        if end_anywhere:
            return [0, 2**idx, 2**idx]  # [X-node, next-B-node, both]
        else:
            return [2**idx]  # [next-B-node]
    if end_anywhere and all_components:
        # All possible non-empty combinations of X-node, next-B-node, and next-O-node
        return [0, 2**(idx-1), 2**(idx), 2**(idx-1), 2**(idx), 2**(idx-1)|2**(idx), 2**(idx-1)|2**(idx)]
    else:
        # All possible non-empty combinations of next-B-node and next-O-node
        return [2**(idx-1), 2**(idx), 2**(idx-1)|2**(idx)]

def build_transition_matrix(level, start_anywhere=False, end_anywhere=False, all_components=False):
    num_states = 2**level
    result = [[0]*num_states for i in range(num_states)]  # The transition matrix
    for state in range(num_states):
        nodes = to_bin(state, level)
        next_states = []
        # For each active node, get the next states
        for idx, node in enumerate(reversed(nodes)):
            if node == 0:
                continue
            next_states.append(get_next_states(idx, end_anywhere, all_components))
        if start_anywhere:
            next_states.append([0, 2**(level-1)])  # Any top B-node can be activated
        for next_nodes in product(*next_states):
            next_state = 0
            for next_node in next_nodes:
                next_state |= next_node
            result[state][next_state] += 1
    return np.array(result, dtype=np.int64)

def count_full(n, level=3, start_anywhere=False, end_anywhere=False, all_components=False, verbose=False):
    mul_mat = build_transition_matrix(level, start_anywhere, end_anywhere, all_components)
    if verbose:
        print('Transition matrix:')
        print_matrix(mul_mat)
        num_edges = mul_mat.sum(axis=1).sum(axis=0)
        print('Number of edges for each transition: {}'.format(num_edges))
        print('Number of edges total: {}'.format(num_edges*n))
    vec = [0]*(2**level)  # Create one state for each node combination
    vec[0] = 1  # Base case
    vec[1] = 1  # The bottom right node is always connected to an X-node
    if all_components:
        # Any combination of B-nodes (even-indexed nodes) is a possible starting point
        num_components = (level+1)//2
        for i in range(1, 2**num_components):
            nodes = to_bin(i, num_components)
            mask = 0
            for idx, node in enumerate(reversed(nodes)):
                if node == 0:
                    continue
                mask |= 2**(2*idx)
            vec[mask] = 1
    vec = np.array(vec, dtype=np.int64)
    for i in range(1, n):
        vec = mul_mat.dot(vec)  # Do matrix multiplication to get next iteration
        if not all_components:  # Only the bottom layer is connected to X-nodes
            if i < level:  # When it is impossible for the upper B-nodes to reach the X-nodes
                for state in range(2**(i+1), 2**(level)):  # The upper B-nodes must not activate
                    vec[state] = 0
        # pprint(vec)
    if start_anywhere:
        return vec[0]+vec[2**(level-1)]
    else:
        return vec[2**(level-1)]

def main():
    parser = argparse.ArgumentParser('count_graph.py')
    parser.add_argument('-n', dest='n', type=int, default=4, help=('The length of the sequence'))
    parser.add_argument('-k', '--n_components', dest='n_components', type=int, default=2, help=('The number of components'))
    parser.add_argument('-s', '--start_anywhere', dest='start_anywhere', action='store_true', default=False, help=('Whether the graph can start anywhere'))
    parser.add_argument('-e', '--end_anywhere', dest='end_anywhere', action='store_true', default=False, help=('Whether the graph can end anywhere'))
    parser.add_argument('-a', '--all_components', dest='all_components', action='store_true', default=False, help=('Whether the graph contains all possible number of components'))
    parser.add_argument('--split', dest='split', action='store_true', default=False, help=('Whether the graph represents the Split model'))
    parser.add_argument('--shared', dest='shared', action='store_true', default=False, help=('Whether the graph represents the Shared model'))
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False, help=('Whether to print more info'))
    parsed = parser.parse_args(sys.argv[1:])
    n = parsed.n
    n_components = parsed.n_components
    start_anywhere = parsed.start_anywhere
    end_anywhere = parsed.end_anywhere
    all_components = parsed.all_components
    verbose = parsed.verbose
    if parsed.split:
        result = 1
        for n_comp in range(1, n_components+1):
            result *= count_full(n, 2*n_comp-1, True, True, False, verbose)
        print(result)
    elif parsed.shared:
        result = count_full(n, 2*n_components-1, True, True, True, verbose)
        print(result)
    else:
        print(count_full(n, 2*n_components-1, start_anywhere, end_anywhere, all_components, verbose))

    test_cases = []
    # One component
    test_cases.append((1, 1, 1, False, False, False))
    test_cases.append((2, 1, 1, True, False, False))
    test_cases.append((1, 1, 1, False, True, False))
    test_cases.append((2, 1, 1, True, True, False))
    # Two components
    test_cases.append((0, 1, 2, False, False, False))
    # test_cases.append((0, 1, 2, True, False, False))
    test_cases.append((0, 1, 2, False, True, False))
    # test_cases.append((0, 1, 2, True, True, False))
    test_cases.append((1, 1, 2, False, False, True))
    test_cases.append((2, 1, 2, True, False, True))
    test_cases.append((1, 1, 2, False, True, True))
    test_cases.append((2, 1, 2, True, True, True))

    test_cases.append((0, 2, 2, False, False, False))
    # test_cases.append((0, 2, 2, True, False, False))
    test_cases.append((0, 2, 2, False, True, False))
    # test_cases.append((0, 2, 2, True, True, False))
    test_cases.append((1, 2, 2, False, False, True))
    test_cases.append((4, 2, 2, True, False, True))
    test_cases.append((3, 2, 2, False, True, True))
    test_cases.append((8, 2, 2, True, True, True))

    test_cases.append((1, 3, 2, False, False, False))
    test_cases.append((2, 3, 2, True, False, False))
    test_cases.append((1, 3, 2, False, True, False))
    test_cases.append((2, 3, 2, True, True, False))
    test_cases.append((3, 3, 2, False, False, True))
    test_cases.append((16, 3, 2, True, False, True))
    test_cases.append((15, 3, 2, False, True, True))
    test_cases.append((80, 3, 2, True, True, True))

    test_cases.append((80, 3, 3, True, True, True))

    for test_case in test_cases:
        gold, n, n_components, start_anywhere, end_anywhere, all_components = test_case
        count = count_full(n, 2*n_components-1, start_anywhere, end_anywhere, all_components)
        if count != gold:
            print('{} != {} for n={}, n_components={}, start_anywhere={}, end_anywhere={}, all_components={}'.format(count, gold, n, n_components, start_anywhere, end_anywhere, all_components))

if __name__ == '__main__':
    main()

