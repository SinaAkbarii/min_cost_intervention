from copy import copy
import networkx as nx
from utils.core_solvers import solveHittingSet


def Alg2(g, S, hittingSetExactSolver=True): # Exact (or Approx.) Algorithm
    """
    Solve minimum-cost intervention through Algorithm 2 (refer to the paper)
    :param g: an instance of ADMG
    :param S: set of nodes corresponding to the query Q[S]
    :param hittingSetExactSolver: if True, the required minimum hitting set problem is solved exactly. Otherwise a
    greedy approximation is provided, which guarantees a log-factor approximation.
    :return: the optimal intervention set, its cost, and the number of hedges discovered throughout the algorithm.
    """
    comps = nx.connected_components(g.nodeSubgraph(S, directed=False))  # C-components of S
    comps = [c for c in comps]
    dirP = []
    for comp in comps:
        dirP += list(g.directParents(comp))
    dirP = set(dirP)
    h = copy(g)  # not to change anything in g
    h.permIntervene(dirP)  # permanently intervene on direct parents. We need them any way
    unid_comps = [c for c in comps if not h.isIdentifiable(c)]
    if len(unid_comps) == 0:  # identifiability already achieved.
        return [dirP, g.interventionCost(dirP), 0]
    H_init = {tuple(c): h.hedgeHull(c) for c in unid_comps}  # hedge hulls of each component
    H_hitset = copy(H_init)
    H = copy(H_hitset)
    H_uni = set.union(*H.values())
    hedgesList = []  # will keep track of all the discovered hedges.
    appearances = {u: [c for c in unid_comps if u in H[tuple(c)]] for u in H_uni.difference(S)}  # which node
    # of H appears in the hedge hull of which nodes of S
    identified_comp = {tuple(c): False for c in unid_comps}
    while True:
        tempInterventionSet = []
        # hedge discovery begins:
        H = copy(H_hitset)
        while True:  # At the end of this while loop, we have an intervention set (temp) which makes Q[S] identifiable.
            while True:  # intervene on one variable greedily until Q[S] becomes identifiable.
                greedyGain = {u: len(appearances[u]) / g.interventionCost({u}) for u in appearances.keys()}  # gain of
                # intervention on a node of H
                xtoIntervene = max(greedyGain, key=greedyGain.get)
                for c in appearances[xtoIntervene]:
                    if not identified_comp[tuple(c)]:
                        temp_hhull = h.hedgeHull(c, H[tuple(c)].difference([xtoIntervene]))
                        for v in H[tuple(c)].difference(S.union(temp_hhull)):
                            appearances[v].remove(c)
                        if temp_hhull == c:
                            identified_comp[tuple(c)] = True  # Q[c] is identified
                            tempInterventionSet.append(xtoIntervene)
                            hedgesList.append(H[tuple(c)].difference(S))   # smallest discovered hedge for Q[c]
                        else:
                            H[tuple(c)] = temp_hhull
                if all(identified_comp.values()):
                    break
            H = {tuple(c): h.hedgeHull(c, H_hitset[tuple(c)].difference(tempInterventionSet)) for c in unid_comps}
            identified_comp = {tuple(c): H[tuple(c)] == c for c in unid_comps}
            if all(identified_comp.values()):
                break
            appearances = {u: [c for c in unid_comps if u in H[tuple(c)]] for u in H_uni.difference(S)}
        # a round of hedge discovery is done. we solve the minimum hitting set for all the hedges already found.
        hittingSetSolution = solveHittingSet(list(h.get_nodes()), hedgesList, h.get_nodeCosts(),
                                             exact=hittingSetExactSolver)
        H_hitset = {tuple(c): h.hedgeHull(c, H_init[tuple(c)].difference(hittingSetSolution)) for c in unid_comps}
        identified_comp = {tuple(c): H_hitset[tuple(c)] == c for c in unid_comps}
        if all(identified_comp.values()):  # Problem solved, all hedges hit
            break
        appearances = {u: [c for c in unid_comps if u in H_hitset[tuple(c)]] for u in H_uni.difference(S)}
    I = dirP.union(hittingSetSolution)
    return [I, g.interventionCost(I), len(hedgesList)]