import networkx as nx
import numpy as np

from camovox import ports, reduce


def _random_tree_metric(rng, n_ports=7):
    # Random binary-ish tree: ports are leaves 0..n-1, internal nodes after.
    g = nx.Graph()
    nodes = list(range(n_ports))
    nxt = n_ports
    while len(nodes) > 2:
        a, b = rng.choice(len(nodes), 2, replace=False)
        u, v = nodes[a], nodes[b]
        g.add_edge(u, nxt, w=rng.uniform(1, 5))
        g.add_edge(v, nxt, w=rng.uniform(1, 5))
        nodes = [x for k, x in enumerate(nodes) if k not in (a, b)] + [nxt]
        nxt += 1
    g.add_edge(nodes[0], nodes[1], w=rng.uniform(1, 5))
    lengths = dict(nx.all_pairs_dijkstra_path_length(g, weight="w"))
    return np.array([[lengths[i][j] for j in range(n_ports)] for i in range(n_ports)])


def test_fit_tree_recovers_an_exact_tree_metric(rng):
    d = _random_tree_metric(rng)
    tree = reduce.fit_tree(d)
    assert tree.fit_error < 1e-8
    np.testing.assert_allclose(tree.distances(), d, atol=1e-8)


def test_contract_tree_merges_short_branches(rng):
    tree = reduce.fit_tree(_random_tree_metric(rng))
    small = reduce.contract_tree(tree, min_resistance=np.median(tree.resistance))
    assert small.n_nodes < tree.n_nodes
    assert len(small.port_node) == len(tree.port_node)


def test_diag_lowrank_bound_is_zero_for_diag_plus_rank2(rng):
    low = rng.normal(size=(8, 2))
    z = low @ low.T + np.diag(rng.uniform(0.5, 1.0, 8))
    err = ports.diag_lowrank_bound(z)
    assert err[2] < 1e-3
    assert np.all(err <= ports.rank_bound(z) + 1e-9)
