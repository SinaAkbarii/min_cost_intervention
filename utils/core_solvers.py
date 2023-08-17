import numpy as np
import itertools
from copy import copy
import networkx as nx


def solveHittingSet(U, T, costs, exact=False):
    """
    Solves the minimum hitting set problem.
    :param U: the universe.
    :param T: a set family. the set of subsets of U that need to be 'hit'.
    :param costs: a dict mapping each member of the universe (U) to its corresponding cost.
    :param exact: if True, the minimum hitting set instance is solved exactly, in a brute-force fashion. Otherwise a
    greedy approximation is found.
    :return: Minimum hitting set, a subset of U.
    """
    if exact:
        setUnion = set.union(*(t for t in T))
        sorted_members, sorted_costs = (np.array(t) for t in zip(*sorted({m: costs[m] for m in setUnion}.items(),
                                                                         key=lambda item: item[1])))
        minCost = np.inf
        for i in range(0, len(setUnion) + 1):  # all possible subsets
            for sub in itertools.combinations(range(len(setUnion)), i):
                subset = set(sorted_members[list(sub)])
                subCost = np.sum(sorted_costs[list(sub)])
                if subCost < minCost:
                    doesHit = True
                    for t in T:
                        if len(subset.intersection(t)) == 0:
                            doesHit = False
                            break
                    if doesHit:
                        hittingSet = subset
                        minCost = subCost
            if minCost < sum(sorted_costs[:i + 1]):
                break
    else:  # Greedy Approximation Algorithm
        if len(T) == 1:
            costsT = {t: costs[t] for t in T[0]}
            return {min(costsT, key=costsT.get)}
        hittingSet = []
        appearances = {u: [] for u in U}
        for i in range(len(T)):
            seti = T[i]
            for x in seti:
                appearances[x].append(i)
        num_sets = len(T)
        while num_sets > 0:
            appearanceNumber = {u: len(appearances[u])/costs[u] for u in U}
            xtoAdd = max(appearanceNumber, key=appearanceNumber.get)
            hittingSet.append(xtoAdd)
            num_sets -= len(appearances[xtoAdd])
            for i in copy(appearances[xtoAdd]):
                seti = T[i]
                for y in seti:
                    appearances[y].remove(i)
    return hittingSet


def solveMinVertexCut(g, source, target, forbidden):
    """
    Minimum vertex cut between two sets of variables. Solves both on directed and bidirected graphs.
    IMPORTANT NOTE: Source nodes CAN be included in the cut, unless included in forbidden!
    this is for our purposes of min-cost intervention, where intervention on source nodes is possible (but not
    target nodes, which will be included in forbidden.)
    :param g: a networkx graph (either Graph or Digraph),
    :param source: a set of nodes of g.
    :param target: a set of nodes of g.
    :param forbidden: a set of forbidden nodes (including, but not limited to target nodes)
    :return: a list of min-vertex-cut between the source and target.
    """
    directed = nx.is_directed(g)
    weights = dict(g.nodes(data='weight', default=np.inf))
    # make sure we do not include S nodes in the min-cut:
    for f in forbidden:
        weights[f] = np.inf
    # transform vertex-cut to edge-cut:
    h = nx.DiGraph()
    for v in g.nodes:
        h.add_edge(str(v) + "/1", str(v) + "/2", capacity=weights[v])
        for w in g.adj[v]:  # successors of v in g:
            h.add_edge(str(v) + "/2", str(w) + "/1", capacity=np.inf)
    # add a node and connect it to the source nodes:
    for v in source:
        h.add_edge("x_source", str(v) + "/1", capacity=np.inf)
        if not directed:
            h.add_edge(str(v) + "/2", "x_source", capacity=np.inf)
    # add a node and connect all target nodes to it:
    for t in target:
        h.add_edge(str(t) + "/2", "y_target", capacity=np.inf)
        if not directed:
            h.add_edge("y_target", str(t) + "/1", capacity=np.inf)

    # the graph is constructed. solve the min-cut:
    try:
        _, partition = nx.minimum_cut(h, "x_source", "y_target")
    except nx.NetworkXUnbounded:
        return 'inf'
    reachable, non_reachable = partition

    # take the smaller of the reachable and non_reachable:
    if len(reachable) < len(non_reachable):
        part = list(reachable)
        part.remove("x_source")
    else:
        part = list(non_reachable)
        part.remove("y_target")

    # find the edges in the cut (representing the nodes)
    node_list = [v.split('/')[0] for v in part]
    return [v for v in node_list if node_list.count(v) == 1]