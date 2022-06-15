import itertools
import networkx as nx
import warnings
from collections import deque
from copy import copy
import numpy as np
from matplotlib import pyplot as plt

warningsOn = True

"""" Extending the Networkx functions to support multiple nodes as the bfs source:
 Note: The following functions are modified based on Networkx implementations for the purposes of this work, to allow
 for finding the set of ancestors of a set of nodes rather than only a single node.
"""


def generic_bfs_edges_general(G, source, neighbors=None, depth_limit=None, sort_neighbors=None):
    if callable(sort_neighbors):
        _neighbors = neighbors
        neighbors = lambda node: iter(sort_neighbors(_neighbors(node)))

    visited = {node for node in source}
    if depth_limit is None:
        depth_limit = len(G)
    queue = deque([(node, depth_limit, neighbors(node)) for node in source])
    while queue:
        parent, depth_now, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                yield parent, child
                visited.add(child)
                if depth_now > 1:
                    queue.append((child, depth_now - 1, neighbors(child)))
        except StopIteration:
            queue.popleft()


def bfs_edges_general(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    if reverse and G.is_directed():
        successors = G.predecessors
    else:
        successors = G.neighbors
    yield from generic_bfs_edges_general(G, source, successors, depth_limit, sort_neighbors)


# source in the input of ancestors_general is a set of nodes (a list, or a set)
# IMPORTANT NOTE: ancestors_general() includes the source itself unlike nx.ancestors()
def ancestors_general(G, source):
    return {child for parent, child in bfs_edges_general(G, source, reverse=True)}.union(source)


class ADMG:
    # initialize the admg instance with g_dir (directed graph) and g_bi (bidirected graph) over the same set of nodes.
    def __init__(self, g_dir, g_bi):
        if warningsOn:
            if g_dir.nodes != g_bi.nodes:
                warnings.warn('Mismatched node names/ Exiting.')
                return
        self.n = len(g_dir.nodes)  # number of nodes
        self.g_dir = g_dir
        self.g_bi = g_bi
        self.nodes = set(g_dir.nodes)
        self.nodeCosts = dict(g_dir.nodes(data='weight', default=np.inf))
        return

    # return the set of nodes:
    def get_nodes(self):
        return self.nodes

    # return the nodeCosts:
    def get_nodeCosts(self):
        return self.nodeCosts

    # Is a set of nodes ancestral for S?
    def isAncestral(self, S, H):
        # consider the subgraph over H:
        sub_g = nx.subgraph(self.g_dir, H)
        return H == ancestors_general(sub_g, S)

    # return the parents of S which are not in S itself:
    def parents(self, S, H=None):
        if H is None:
            dag = self.g_dir
        else:
            dag = nx.subgraph(self.g_dir, H)
        parS = []
        for s in S:
            parS += dag.predecessors(s)
        return set(parS).difference(set(S))

    # return those nodes that have a bidirected edge to at least one node in S:
    def bidir(self, S, H=None):
        if H is None:
            g = self.g_bi
        else:
            g = nx.subgraph(self.g_bi, H)
        bidirS = []
        for s in S:
            bidirS += g.neighbors(s)
        return set(bidirS).difference(set(S))

    # return the intersection of bidir and parents of S:
    def directParents(self, S, H=None):  # directParents must be intervened upon.
        return self.parents(S, H).intersection(self.bidir(S, H))

    # does a subgraph H form a hedge for S?
    def isHedge(self, S, H):
        if warningsOn:
            if not set(S).issubset(H):
                warnings.warn("Call to isHedge: S is not a subset of H!")
            if not nx.is_connected(nx.subgraph(self.latent, S)):
                warnings.warn("Call to isHedge: S is not a c-component!")

        # H forms a hedge for S iff it is ancestral and it is a c-component:
        if self.isAncestral(S, H):
            if nx.is_connected(nx.subgraph(self.g_bi, H)):
                return True
        return False

    # Construct the hedge hull of S in the subgraph H:
    def hedgeHull(self, S, H=None):
        if H is None:
            H = self.nodes  # The whole graph
        if warningsOn:
            if not S.issubset(H):
                warnings.warn("Call to hedgeHull: S is not a subset of H!")
                return None
            if not nx.is_connected(nx.subgraph(self.g_bi, S)):
                warnings.warn("Call to hedgeHull: S is not a c-component!")
                return None
        subset = copy(H)
        # Ancestor of S in H:
        anc_set = ancestors_general(nx.subgraph(self.g_dir, subset), S)
        s = list(S)[0]
        # connected component of S in anc_set:
        con_comp = nx.node_connected_component(nx.subgraph(self.g_bi, anc_set), s)
        if con_comp == subset:
            return subset
        subset = con_comp
        # Find the largest set of nodes which is ancestral for S and is a c-component:
        while True:
            anc_set = ancestors_general(nx.subgraph(self.g_dir, subset), S)
            if anc_set == subset:
                return subset
            subset = anc_set
            con_comp = nx.node_connected_component(nx.subgraph(self.g_bi, subset), s)
            if con_comp == subset:
                return subset
            subset = con_comp

    # Determine if S is identifiable in subgraph H
    def isIdentifiable(self, S, H=None):
        if H is None:
            H = self.nodes  # The whole graph
        if warningsOn:
            if not set(S).issubset(H):
                # S is not a subset of H, so not ID
                return False
        if set(S) == self.hedgeHull(S, H):
            return True
        return False

    # calculate the cost of intervention on a set I:
    def interventionCost(self, I={}):
        return sum([self.nodeCosts[i] for i in I])

    # check whether Q[S] becomes identifiable after intervention on I. return the cost of this intervention as well.
    def interventionResult(self, S, I={}):
        H = set(self.nodes).difference(I)  # intervention on I is equivalent to looking at Q[H]
        return [self.isIdentifiable(S, H), self.interventionCost(I)]

    # brute force algorithm to determine the optimal intervention to identify Q[S]:
    def optimalIntervention(self, S, H=None):
        if H is None:
            H = copy(self.nodes)
        comps = nx.connected_components(self.nodeSubgraph(S, directed=False))  # C-components of S
        comps = [c for c in comps]
        dirParents = []
        for comp in comps:
            dirParents += list(self.directParents(comp, H))
        dirParents = set(dirParents)
        # dirParents must be intervened upon.
        baseCost = sum([self.nodeCosts[i] for i in dirParents])
        if all([self.interventionResult(c, dirParents)[0] for c in comps]):  # if the set of direct Parents
            # is enough to identify
            return [dirParents, baseCost]
        HHulls = {tuple(c): self.hedgeHull(c, H.difference(dirParents)) for c in comps}  # hedge hulls of each component
        H_uni = set.union(*HHulls.values())
        minCostAdd = np.inf
        optInterv = H_uni.difference(S)
        optCosts = sorted(copy([self.nodeCosts[v] for v in optInterv]))
        for i in range(0, len(optInterv) + 1):  # all subsets
            for subset in itertools.combinations(optInterv, i):
                if sum([self.nodeCosts[i] for i in subset]) < minCostAdd:
                    I = set(subset).union(dirParents)
                    if all([self.interventionResult(c, I)[0] for c in comps]):
                        minCostAdd = sum([self.nodeCosts[i] for i in subset])
                        intervSet = I
            if minCostAdd < sum(optCosts[:i + 1]):
                break
        return [intervSet, baseCost + minCostAdd]

    # return the node with the smallest cost to intervene upon.
    def smallestCostVertex(self, H):  # return the min cost vertex among H
        costsH = {v: self.nodeCosts[v] for v in H}
        return min(costsH, key=costsH.get)

    # permanently intervene on a set I of nodes.
    def permIntervene(self, I={}):
        self.nodes = self.nodes.difference(I)
        self.g_bi = nx.subgraph(self.g_bi, self.nodes)
        self.g_dir = nx.subgraph(self.g_dir, self.nodes)
        self.nodeCosts = {v: self.nodeCosts[v] for v in self.nodes}
        return

    # return a particular subgraph over directed or bidirected edges only on the set of nodes H:
    def nodeSubgraph(self, H, directed=True):
        if directed:
            return nx.subgraph(self.g_dir, H)
        else:
            return nx.subgraph(self.g_bi, H)

    # count the number of hedges formed for Q[S] in H:
    def countHedges(self, S, H=None):  # Count the number of hedges formed for S in H
        if H is None:
            H = copy(self.nodes)
        count = 0
        if S.issubset(H):
            H = self.hedgeHull(S, H)
            if H == S:
                return 0
            HminusS = list(H.difference(S))
            count += self.countHedges(S, H.difference([HminusS[0]]))
            for i in range(1, len(HminusS)):  # all subsets
                for subset in itertools.combinations(list(set(HminusS).difference([HminusS[0]])), i):
                    I = list(set(subset).union(S).union([HminusS[0]]))
                    if self.isHedge(S, I):
                        count += 1
        else:
            warnings.warn('Call to countHedges: S is not a subset of H!')
        return count

    # plot the admg over the nodes H for the causal query Q[S]
    def plotWithNodeWeights(self, S={}, H=None):
        if H is None:
            H = set(self.g_dir.nodes)   # The whole graph
        G1 = self.nodeSubgraph(H, directed=True)
        G2_edges = self.nodeSubgraph(H, directed=False).edges
        G2 = nx.DiGraph()
        G2.add_edges_from(G2_edges)
        pos = nx.kamada_kawai_layout(G1)
        nx.draw_networkx_nodes(G1, pos, node_size=400,
                               nodelist=list(H.difference(S)),
                               node_color=None)
        nx.draw_networkx_nodes(G1, pos, node_size=400, nodelist=list(S), node_color='red')
        nx.draw_networkx_edges(G1, pos)
        nx.draw_networkx_edges(G2, pos, style=':', connectionstyle="arc3, rad=-0.4", arrowsize=0.01, edge_color='blue')
        nx.draw_networkx_labels(G1, pos)
        for i in H:
            x, y = pos[i]
            plt.text(x+0.03, y+0.03, s=str(self.nodeCosts[i]), bbox=dict(facecolor='yellow', alpha=0.5),
                     horizontalalignment='center', fontsize=7)
        plt.show()
        return






# a function to solve the minimum hitting set problem.
# exact approach is a brute-force to check every possible combination,
# approx approach is the greedy algorithm choosing the node maximising the marginal gain at each iteration.
# The universe is U, T is the list of sets.
# costs is a dict mapping each member of the universe to its corresponding cost.
def solveHittingSet(U, T, costs, exact=False):
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


# Solve minimum-cost intervention through Algorithm 2 (min-cost paper)
# input arguments: g is an admg with weights on nodes, S is the query (as in Q[S])
# third argument decides whether we want to solve it exactly or with a greedy algorithm with logarithmic factor approx.
# returns the optimal intervention set, its cost, and the number of hedges discovered throughout the algorithm.
def Alg2(g, S, hittingSetExactSolver=True):  # Exact (or Approx.) Algorithm
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


# minimum vertex cut between two sets of variables. Solves both on directed and bidirected graphs.
# g is a networkx graph (either Graph or Digraph),
# source and target are a set of nodes of g.
# returns a list of min-vertex-cut between the source and target.
# IMPORTANT NOTE: Source nodes CAN be included in the cut, unless included in forbidden!
# this is for our purposes of min-cost intervention, where intervention on source nodes is possible (but not
# target nodes, which will be included in forbidden.)
# forbidden is a set of forbidden nodes (including, but not limited to target nodes)
def solveMinVertexCut(g, source, target, forbidden):
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


# a heuristic post-process to reduce the cost of heuristic algorithms:
# g is an ADMG, S is the query as in Q[S],
# A is a list containing the output of a heuristic alg.
def heuristicPostProcess(g, S, A):      # make A smaller as long as Q[S] is identifiable
    weights = g.get_nodeCosts()
    comps = nx.connected_components(g.nodeSubgraph(S, directed=False))  # C-components of S
    comps = [c for c in comps]
    A, _ = (list(x) for x in zip(*sorted({a: weights[a] for a in A}.items(), key=lambda item: item[1])))
    V = list(set(g.nodes).difference(A))
    for a in A:
        if all(g.isIdentifiable(c, set(V+[a])) for c in comps):
            V += [a]
    return set(g.nodes).difference(V)


# Min-cut based heuristic algorithm for minimum-cost intervention
# receives an instance of ADMG with node weights along with the query Q[S]
# returns a set which is sufficient to intervene upon for identification of Q[S], along with its cost.
# followed by an optional heuristic post-process to reduce the cost
def heuristicMinCut2(g, S, postProcess=True):     # break the ancestral sets where bidir(S) are present.
    print(S)
    comps = nx.connected_components(g.nodeSubgraph(S, directed=False))  # C-components of S
    comps = [c for c in comps]
    dirP = []
    for comp in comps:
        dirP += list(g.directParents(comp))
    dirP = set(dirP)
    h = copy(g)
    h.permIntervene(dirP)  # permanently intervene on direct parents. We need them any way
    unid_comps = [c for c in comps if not h.isIdentifiable(c)]
    if len(unid_comps) == 0:    # dirP is enough for the identification of Q[S]
        return [dirP, g.interventionCost(dirP)]
    S_unid = set.union(*unid_comps)
    H = set.union(*[h.hedgeHull(c) for c in unid_comps])
    # Construct the graph \mathcal{H}:
    dirSubg = h.nodeSubgraph(H, directed=True)
    bidirS = h.bidir(S_unid, H)
    minCut = solveMinVertexCut(dirSubg, bidirS, S_unid, S)
    if minCut == 'inf':
        print('could not find a solution. returning infinite cost.')
        return [[], np.inf]
    if postProcess:
        minCut = heuristicPostProcess(h, S_unid, minCut)
    I = dirP.union(minCut)
    return [I, g.interventionCost(I)]


# Min-cut based heuristic algorithm for minimum-cost intervention
# receives an instance of ADMG with node weights along with the query Q[S]
# returns a set which is sufficient to intervene upon for identification of Q[S], along with its cost.
# followed by an optional heuristic post-process to reduce the cost
def heuristicMinCut1(g, S, postProcess=True):     # break the c-components where pa(S) are present.
    comps = nx.connected_components(g.nodeSubgraph(S, directed=False))  # C-components of S
    comps = [c for c in comps]
    dirP = []
    for comp in comps:
        dirP += list(g.directParents(comp))
    dirP = set(dirP)
    h = copy(g)
    h.permIntervene(dirP)  # permanently intervene on direct parents. We need them any way
    unid_comps = [c for c in comps if not h.isIdentifiable(c)]
    if len(unid_comps) == 0:  # dirP is enough for the identification of Q[S]
        return [dirP, g.interventionCost(dirP)]
    S_unid = set.union(*unid_comps)
    H = set.union(*[h.hedgeHull(c) for c in unid_comps])
    # Construct the graph \mathcal{H}:
    biSubg = h.nodeSubgraph(H, directed=False)  # bidirected subgraph
    parentS = h.parents(S_unid, H)
    minCut = solveMinVertexCut(biSubg, parentS, S_unid, S)
    if minCut == 'inf':
        print('could not find a solution. returning infinite cost.')
        return [[], np.inf]
    if postProcess:
        minCut = heuristicPostProcess(h, S_unid, minCut)
    I = dirP.union(minCut)
    return [I, g.interventionCost(I)]


# Naive greedy algorithm for minimum cost intervention.
# intervenes upon nodes greedily until Q[S] becomes identifiable.
# receives an ADMG g, query Q[S], and is followed by an optional post-process to reduce the cost.
def naiveGreedy(g, S, postProcess=True):  # Reduce the sum of the cost of the remaining hedge hull
    comps = nx.connected_components(g.nodeSubgraph(S, directed=False))  # C-components of S
    comps = [c for c in comps]
    dirP = []
    for comp in comps:
        dirP += list(g.directParents(comp))
    dirP = set(dirP)
    h = copy(g)
    h.permIntervene(dirP)  # permanently intervene on direct parents. We need them any way
    unid_comps = [c for c in comps if not h.isIdentifiable(c)]
    if len(unid_comps) == 0:  # identifiability already achieved.
        return [dirP, g.interventionCost(dirP), 0]
    H = {tuple(c): h.hedgeHull(c) for c in unid_comps}  # hedge hulls of each component
    H_uni = set.union(*H.values())
    appearances = {u: [c for c in unid_comps if u in H[tuple(c)]] for u in H_uni.difference(S)}  # which node
    I = []
    identified_comp = {tuple(c): False for c in unid_comps}
    while True:
        greedyGain = {u: len(appearances[u]) / g.interventionCost({u}) for u in appearances.keys()}  # gain of
        xtoIntervene = max(greedyGain, key=greedyGain.get)
        I.append(xtoIntervene)
        for c in appearances[xtoIntervene]:
            if not identified_comp[tuple(c)]:
                new_hh = h.hedgeHull(c, H[tuple(c)].difference([xtoIntervene]))
                for v in H[tuple(c)].difference(new_hh).difference(S):
                    appearances[v].remove(c)
                if new_hh == c:
                    identified_comp[tuple(c)] = True
                else:
                    H[tuple(c)] = new_hh
        if all(identified_comp.values()):
            break
    if postProcess:
        I = heuristicPostProcess(h, S, I)
    I = dirP.union(I)
    return [I, g.interventionCost(I)]



if __name__ == '__main__':

    # build a DiGraph and a Graph:
    g_dir = nx.DiGraph()  # the directed component of the ADMG
    g_bi = nx.Graph()  # the bidirected component of the ADMG
    nodes = [('a', {'weight': 5}), ('2', {'weight': 4}), ('3', {'weight': 3}), ('4', {'weight': 5}),
             ('5', {'weight': 1})]
    g_dir.add_nodes_from(nodes)
    g_bi.add_nodes_from(nodes)
    dir_edges = [('a', '2'), ('3', '2'), ('4', 'a'),
                 ('5', '3'), ('4', '5')]
    bi_edges = [('a', '4'), ('5', '4'), ('4', '2'),
                ('4', '3')]
    g_dir.add_edges_from(dir_edges)
    g_bi.add_edges_from(bi_edges)

    # build an instance of ADMG with g_dir and g_bi:
    g = ADMG(g_dir, g_bi)
    Y = {'2'}


    intervention, cost = heuristicMinCut2(g, Y)       # heuristic min-cut based algorithm 2 for min-cost intervention design
    print('heuristicMinCut2: [' + str(intervention) + ', ' + str(cost) + ']')
    intervention, cost = heuristicMinCut1(g, Y)        # heuristic min-cut based algorithm 1 for min-cost
    print('heuristicMinCut1: [' + str(intervention) + ', ' + str(cost) + ']')
    intervention, cost = naiveGreedy(g, Y, postProcess=True)            # a naive greedy for min-cost intervention
    print('Naive Greedy: [' + str(intervention) + ', ' + str(cost) + ']')
    intervention, cost = h.optimalIntervention(Y)  # brute-force algorithm for min-cost intervention which checks
    # for every subset
    print('Brute force: [' + str(intervention) + ', ' + str(cost) + ']')
    intervention, cost, _ = Alg2(g, Y)              # minimum-hitting set based more efficient algorithm for
    # min-cost intervention
    print('Exact Alg2: [' + str(intervention) + ', ' + str(cost) + ']')
    intervention, cost, _ = Alg2(g, Y, hittingSetExactSolver=False)     # same algorithm, but solves the
    # min-hitting set approximately
    print('Approx. Alg2: [' + str(intervention) + ', ' + str(cost) + ']')
