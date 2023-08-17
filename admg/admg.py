import networkx as nx
from utils.bfs_modif import ancestors_general
from copy import copy
import itertools
import warnings
from matplotlib import pyplot as plt
import numpy as np


class ADMG:
    """
        Main class for ADMGs.
        Each ADMG comprises one nx.DiGraph and one nx.Graph to represent the directed edges and the bidirected edges
        among variables, respectively.
    """
    # initialize the admg instance with g_dir (directed graph) and g_bi (bidirected graph) over the same set of nodes.
    def __init__(self, g_dir, g_bi):
        assert g_dir.nodes == g_bi.nodes
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
        assert set(S).issubset(H)

        # H forms a hedge for S iff it is ancestral and it is a c-component:
        if self.isAncestral(S, H):
            if nx.is_connected(nx.subgraph(self.g_bi, H)):
                return True
        return False

    # Construct the hedge hull of S in the subgraph H:
    def hedgeHull(self, S, H=None):
        if H is None:
            H = self.nodes  # The whole graph
        assert S.issubset(H)
        assert nx.is_connected(nx.subgraph(self.g_bi, S))

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
        assert set(S).issubset(H)

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