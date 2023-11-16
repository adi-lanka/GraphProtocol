import networkx as nx
from collections import defaultdict
from collections import Counter
import numpy as np


def similarity(g1, g2):
    """
    Return: Set similarity (frac of nodes that appear in both graphs to nodes in graphs) between # of nodes in graph
    """
    g1_nodes = set([str(node) for node in g1.nodes()])
    g2_nodes = set([str(node) for node in g2.nodes()])
    return len(g1_nodes.intersection(g2_nodes)) / len(g1_nodes.union(g2_nodes))


def diameter(g1, num_samples=10000):
    """
    Diameter is consistent but biased width of graph
    See: get_path_stat(g1)
    """
    return get_path_stat(g1, num_samples=num_samples)[0]


def avg_path_length(g1, num_samples=10000):
    """
    See: get_path_stat(g1)
    """
    if num_samples is None:
        return np.mean(
            [
                nx.average_shortest_path_length(g1.subgraph(nodes))
                for nodes in nx.connected_components(nx.Graph(g1))
            ]
        )
    else:
        return get_path_stat(g1, num_samples=num_samples)[1]


def get_path_stat(g1, num_samples=10000):
    """
    Monte Carlo approach to get statistics on density of graph

    Return: (diameter,avg_path_len, path)
    Path: fraction of pairs of nodes that have a path between them
    Diameter is consistent but biased: too low because it's trying to approx. a max by taking max of samples
    """
    curr_avg = 0
    i = 0
    nodes = list(g1.nodes())
    diam = 0
    for count in range(num_samples):
        # if count % 100 == 0:
        #     print("Count:", count)
        try:
            short_path = nx.shortest_path(
                g1, np.random.choice(nodes), np.random.choice(nodes)
            )
        except nx.exception.NetworkXNoPath:
            short_path = None
        if short_path is None:
            continue
        else:
            curr_avg = 1 / (i + 1) * len(short_path) + i / (i + 1) * curr_avg
            diam = max(diam, len(short_path))
            i += 1
    return (diam, curr_avg, i / num_samples)


def reciprocity(g1):
    """
    Return: fraction of pairs of edges that go fwd,backward
    """
    return nx.reciprocity(g1)


def distr_degrees(g1):
    """
    Return: Count {degree, frequency}
    """
    graph_degs = [x[1] for x in list(g1.degree())]
    # deg_distr = defaultdict(lambda: 0)
    # for d in graph_degs:
    #     deg_distr[d] += 1

    return Counter(graph_degs)


def sorted_degrees(g1):
    """
    Return: list of nodes sorted by degrees in descending order
    """
    sorted_graph = sorted(list(g1.degree()), key=(lambda x: x[1]), reverse=True)
    return [x[0] for x in sorted_graph]


def drop_top_n_degrees(g1, n):
    """
    Return: subgraph all nodes except top n nodes (by degree)
    """
    return g1.subgraph(sorted_degrees(g1)[n:])


def drop_degree_gt_n(g1, n):
    """
    Return: subgraph all nodes of graph that have degree <n
    """
    return g1.subgraph([x for x in g1.nodes() if g1.degree(x) < n])


def size_SCC(g1):
    """
    Return: list of sizes of SCC
    To be useful turn result into a Counter
    """
    return [len(scc) for scc in nx.strongly_connected_components(g1)]


def size_WCC(g1):
    """
    Return: list of sizes of WCC
    To be useful turn result into a Counter
    """
    return [len(scc) for scc in nx.weakly_connected_components(g1)]


# gets subgraphs induced by every SCC
# can do analaysis on each subgraph (path stats for ex.)
# S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
def subgraph_SCC(g1):
    """
    Return: largest SCC of graph as a copy of a subgraph
    """
    biggest = max(nx.strongly_connected_components(g1), key=len)
    S = g1.subgraph(biggest).copy()
    return S


def subgraph_WCC(g1):
    """
    Return: largest WCC of graph as a copy of a subgraph
    """
    biggest = max(nx.weakly_connected_components(g1), key=len)
    S = g1.subgraph(biggest).copy()
    return S


def path_analysis(g1: nx.DiGraph, scc_g1: nx.Graph, num_iter=10000):
    """
    Breaks graph down to bowtie structure as in Graph Structure of the Web (A. Broder):
        -scc, in-only, out-only, wcc-only (tendrils)

    Return: Tuple of 2 Dictionaries:
        1. Proportion of sampled nodes in each of 4 components: (in, out, tendrils, scc)
        2. Average number of nodes seen from travelling away from SCC: (out_from_in nodes, in_from_out nodes)

    """
    # initialization
    scc_count = 0
    in_count = 0
    out_count = 0
    tendrils_count = 0
    scc_nodes = frozenset(scc_g1.nodes())
    dfs = nx.algorithms.traversal.depth_first_search
    g1_reverse = g1.reverse()
    in_average_out_nodes = 0
    out_average_in_nodes = 0
    # Monte Carlo approach: selecting nodes, getting counts based on selection
    for i in range(num_iter):
        rand_node = np.random.choice(g1.nodes())
        # Scc nodes connected to in,out,scc -> skip analysis
        if rand_node in scc_nodes:
            scc_count += 1
        # if path node->scc, then it's an "in node"
        elif scc_nodes & set(dfs.dfs_preorder_nodes(g1, rand_node)):
            in_count += 1
            num_out_nodes_found = len(
                list(dfs.dfs_preorder_nodes(g1_reverse, rand_node))
            )
            # update weighted avg of # nodes reachable travelling backwards from "in nodes"
            in_average_out_nodes += (
                num_out_nodes_found - in_average_out_nodes
            ) / in_count
        # if path backwards scc <- node, "out node"
        elif scc_nodes & set(dfs.dfs_preorder_nodes(g1_reverse, rand_node)):
            out_count += 1
            num_in_nodes_found = len(list(dfs.dfs_preorder_nodes(g1, rand_node)))
            # update weighted avg of # of nodes reachable travelling forwards from "out nodes"
            out_average_in_nodes += (
                num_in_nodes_found - out_average_in_nodes
            ) / out_count
        # otherwise only in WCC
        else:
            tendrils_count += 1
    # Return: results packaged into dictionaries
    counts = {
        "in": in_count / num_iter,
        "out": out_count / num_iter,
        "tendrils": tendrils_count / num_iter,
        "scc": scc_count / num_iter,
    }
    outward_nodes = {
        "out_from_in_nodes": in_average_out_nodes,
        "in_from_out_nodes": out_average_in_nodes,
    }
    return counts, outward_nodes


def rand_max_flow(g1, num_iter=100, progress=False):
    """
    Monte Carlo approach: repeatedly select 2 random nodes and find the flow between them
    Flow := # of edges that need to be cut to disconnect the nodes
    Makes most sense to call on SCC
    Return: (avg max flow, min max flow)
    Min max flow is biased: too high
    SIDE EFFECT: not copying graph adding capacity to each edge
    """
    avg_max_flow = 0
    min_max_flow = float("Inf")
    nx.set_edge_attributes(g1, 1, "capacity")

    for i in range(1, num_iter + 1):
        rand_node1 = np.random.choice(g1.nodes())
        rand_node2 = np.random.choice(g1.nodes())
        # make sure source != sink
        # code could technically loop infinitely (don't pass in 1 vertex graph!)
        while rand_node1 == rand_node2:
            rand_node2 = np.random.choice(g1.nodes())
        flow_value, flow_dict = nx.maximum_flow(g1, rand_node1, rand_node2)
        # disconnected edges
        if flow_value < 1:
            print(
                "Found disconnected edges, did you mean to call rand_max_flow() on the SCC?"
            )
            continue
        if progress and i % 10 == 0:
            print(i)
        min_max_flow = min(min_max_flow, flow_value)
        avg_max_flow += 1 / i * (flow_value - avg_max_flow)
    return avg_max_flow, min_max_flow


# use users with multiple transactions as source,sink
def influencer_max_flow(g1, num_nodes=10, progress=False):
    """
    Find max flow between all pairs of nodes that are among top num_nodes (source:out-degree, sink:in-degree)
    Flow := # of edges that need to be cut to disconnect the nodes
    Makes most sense to call on SCC
    Return: (avg max flow, min max flow)
    SIDE EFFECT: not copying graph adding capacity to each edge
    """
    avg_max_flow = 0
    min_max_flow = float("Inf")
    nx.set_edge_attributes(g1, 1, "capacity")
    i = 0
    source_nodes = sorted(g1.nodes(), key=g1.out_degree, reverse=True)[:num_nodes]
    sink_nodes = sorted(g1.nodes(), key=g1.in_degree, reverse=True)[:num_nodes]

    for s in source_nodes:
        for t in sink_nodes:
            if s == t:
                continue
            i += 1
            flow_value, flow_dict = nx.maximum_flow(g1, s, t)
            # disconnected edges
            if flow_value < 1:
                print(
                    "Found disconnected edges, did you mean to call rand_max_flow() on the SCC?"
                )
                continue
            if progress and i % 10 == 0:
                print(i)

            min_max_flow = min(min_max_flow, flow_value)
            avg_max_flow += 1 / i * (flow_value - avg_max_flow)
    return avg_max_flow, min_max_flow


def top_scc_drop_high_deg(g1, max_n):
    """
    To do more analysis in future, by removing bias of large degree accounts, this function removes the nodes with highest degree
    Return: list of subgraph of SCC after removing nodes with highest degree
    """
    sccs = []
    for n in range(max_n):
        g = drop_top_n_degrees(g1, n)
        sccs.append(subgraph_SCC(g))
    return sccs


def _main():
    # moonbird = nx.read_gml("Moonbird_Graph.gml")
    # bayc = nx.read_gml("Constructed_Graph_big.gml")
    # azuki = nx.read_gml("Azuki.gml")
    # meebits = nx.read_gml("Meebits.gml")
    # print(similarity(doodle_moon_meebits_mayc, bayc))
    # terraforms = nx.read_gml("Terraforms.gml")
    # all_washed = nx.read_gml("All_Washed.gml")
    mayc = nx.read_gml("MAYC.gml")
    # random_graph = nx.read_gml("Random_graph.gml")
    # print(avg_path_length(nx.Graph(moonbird)))
    # np.random.seed(42)
    # print(get_path_stat(bayc, num_samples=1000))

    # all_graphs = nx.read_gml("All_Graphs.gml")
    # print("Distribution of degrees (all graphs): ", distr_degrees(all_graphs))
    # print("All graphs (SCC): ", Counter(size_SCC(all_graphs)))
    print("mayc (SCC): ", Counter(size_SCC(mayc)))
    # print("All graphs (WCC): ", Counter(size_WCC(all_graphs)))
    print("mayc (WCC): ", Counter(size_WCC(mayc)))
    ######
    # Construct Random Graph
    # random_graph = nx.gnm_random_graph(
    #     all_graphs.number_of_nodes(), all_graphs.number_of_edges(), directed=True
    # )
    # print("Random graph (SCC): ", Counter(size_SCC(random_graph)))
    # print("Random graph (WCC): ", Counter(size_WCC(random_graph)))
    # nx.write_gml(random_graph, "Random_graph.gml")
    ######
    # path statistics
    print(
        "Path statistics: ",
        get_path_stat(subgraph_WCC(mayc), num_samples=10000),
    )
    # path analysis
    print(
        "Path Analysis:",
        path_analysis(
            # WCC here
            subgraph_WCC(mayc),
            subgraph_SCC(mayc),
            num_iter=5000,
        ),
    )


def test():
    # all_graphs = nx.read_gml("All_Graphs.gml")

    mayc = nx.read_gml("MAYC.gml")

    print(rand_max_flow(subgraph_SCC(all_graphs_wash), num_iter=50, progress=False))
    print(
        influencer_max_flow(subgraph_SCC(all_graphs_wash), num_nodes=10, progress=False)
    )
    # print(distr_degrees(all_graphs))
    # address with highest degree
    print(max(all_graphs_wash.degree(), key=lambda x: x[1])[0])


def test2():
    all_graphs = nx.read_gml("All_Graphs.gml")

    print("deg list: ", distr_degrees(all_graphs))
    # g = drop_top_n_degrees(moonbird, 3)
    #### drop nodes
    # g = drop_degree_gt_n(bayc, 100)
    # print("SCC: ", Counter(size_SCC(g)))
    # print("SCCs: ", [g1.number_of_nodes() for g1 in top_scc_drop_high_deg(bayc, 10)])


if __name__ == "__main__":
    _main()
    # test()
    # test2()
