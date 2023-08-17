""""
    Extending the Networkx functions to support multiple nodes as the bfs source:
    Note: The following functions are modified based on Networkx implementations for the purposes of this work, to allow
    for finding the set of ancestors of a set of nodes rather than only a single node.
"""
from collections import deque


def generic_bfs_edges_general(G, source, neighbors=None, depth_limit=None, sort_neighbors=None):
    """
    Iterate over edges in a breadth-first-search.
    :param G: a networkx.Graph or networkx.DiGraph object.
    :param source: source node in the graph G to start BFS from.
    :param neighbors: the neighbors to explore from a node. In general, predecessors for directed graphs, and all
    neighbors for undirected ones.
    :param depth_limit: the depth up to which the BFS is run.
    :param sort_neighbors: a function that specifies the order in which the neighbors to be explored are sorted.
    :return: edges from the BFS.
    """
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


"""
    Allowing BFS for both directed and undirected graphs:
"""


def bfs_edges_general(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    """
    :param G: a networkx.Graph or networkx.DiGraph object.
    :param source: source node in the graph G to start BFS from.
    :param reverse: traverse the graph in reverse direction.
    :param depth_limit: maximum search depth
    :param sort_neighbors: a function that specifies the order in which the neighbors to be explored are sorted.
    :return: edges from the BFS.
    """
    if reverse and G.is_directed():
        successors = G.predecessors
    else:
        successors = G.neighbors
    yield from generic_bfs_edges_general(G, source, successors, depth_limit, sort_neighbors)


def ancestors_general(G, source):
    """
    Returns the set of ancestors of a given set of nodes.
    IMPORTANT NOTE: ancestors_general() includes the source itself unlike nx.ancestors()
    :param G: a networkx.Graph or networkx.DiGraph object.
    :param source: a set or a list of source nodes
    :return: set of ancestors, including the sources themselves.
    """
    return {child for parent, child in bfs_edges_general(G, source, reverse=True)}.union(source)
