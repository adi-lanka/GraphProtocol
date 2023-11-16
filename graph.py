import api_calls
import networkx as nx
import random
import os
import json
import pandas as pd
import numpy as np

# Early stopping constant: if graph exceeds this size stop constructing graph
# To preserve API calls
# Lower Run-time
MAX_GRAPH_SIZE = 30000


def construct_graph(
    contr_addr, checkpt_rate=1000, offset=1000, checkpt_file=None, resume_from_dir=None
):
    """
    Make graph:
        -make a list of wallets of interest
        -iterate through this list and add new wallets to wallets of interest
    contr_addr: Address of the ERC721 Contract that minted the NFT of interest - all the transaction data will involve this contract
    checkpt_rate: How frequently to save checkpoints in case the program prematurely terminates
    offset: Passed to api_calls.py, how many transactions to process at once
    checkpt_file: Path to directory: Where to save checkpoints.  If None this will assign a default value based on contr_addr
    resume_from_dir: If checkpoint has saved in a directory, where to retrieve checkpoints.  None if starting graph from scratch
        -resume_from_dir = None for new graphs
    """

    # initialize with an empty graph
    # first wallets are addresses that initially minted NFT
    if resume_from_dir is None:
        g = nx.DiGraph()
        count = 0
        # every request will be a cache miss
        wallets = api_calls.get_tx_contract(contr_addr, use_cached=False, offset=1000)
    else:
        # open checkpoint directory
        # loading the checkpoint in graph
        g = nx.read_gml(os.path.join(resume_from_dir, "graph.gml"))
        count = g.number_of_nodes()
        # process logged wallets
        with open(os.path.join(resume_from_dir, "wallets.json"), "r") as f:
            wallets = json.load(f)
    # setting up directory to save checkpoints
    if checkpt_file is None and checkpt_rate is not None:
        checkpt_file = os.path.join("graph_checkpts", contr_addr)
    if checkpt_file is not None and not os.path.exists(checkpt_file):
        os.makedirs(checkpt_file)

    # While graph isn't complete
    # Eventually will run out of wallets even if MAX_GRAPH_SIZE = 'Inf'
    while wallets and count < MAX_GRAPH_SIZE:
        wallet = wallets.pop(0)
        # Neighbours: list of wallets that have receieved a NFT transaction from wallet we are looking at
        neighbours = api_calls.get_neighbours(
            wallet, contr_addr, use_cached=False, offset=offset
        )
        for neighbour in neighbours:
            # If never seen wallet, append to list of wallets
            # BFS of graph as we generate it
            if neighbour not in g.nodes():
                wallets.append(neighbour)
            g.add_edge(wallet, neighbour)
        count += 1
        # update can't run <2s because every wallet takes >0.2s to process (API rate limit)
        if count % 10 == 0:
            print("Iteration Number: ", count)
            print("Wallets remaining: ", len(wallets))
        # save periodic checkpoints
        if checkpt_rate is not None and count % checkpt_rate == 0:
            nx.write_gml(g, os.path.join(checkpt_file, "graph.gml"))
            with open(os.path.join(checkpt_file, "wallets.json"), "w") as f:
                # json.dump(wallets, f)
                f.write(json.dumps(wallets))
    # save checkpoint once we have finished with graph
    if checkpt_rate is not None:
        nx.write_gml(g, os.path.join(checkpt_file, "graph.gml"))
        with open(os.path.join(checkpt_file, "wallets.json"), "w") as f:
            # json.dump(wallets, f)
            f.write(json.dumps(wallets))
    return g


# filepath to contract location
# use list from etherscan to remove contracts (addresses)
# "Txhash","ContractAddress","ContractName"
# so contract don't have to be loaded from csv slowly
# contr_addrs = None


# def remove_contracts(g, contr_list):
#     """
#     Some of the wallets involved with transactions are Contracts such as BatchSwap and this function
#     Attempt to remove the Contract owned addresses from a graph
#     Currently doesn't work because the list of Contract owned addresses (csv from EtherScan) is incomplete
#     """
#     global contr_addrs
#     if contr_addrs is None:
#         contr = pd.read_csv(contr_list)
#         contr_addrs = frozenset([str(c).lower() for c in contr["ContractAddress"]])
#     wallet_list = [addr for addr in g.nodes() if str(addr).lower() not in contr_addrs]
#     return g.subgraph(wallet_list).copy()


def _main():
    # contract addresses
    azuki_contr = "0xED5AF388653567Af2F388E6224dC7C4b3241C544".lower()
   
    #############
    # checkpoint directories

    azuki_checkpt_dir = os.path.join("graph_checkpts", azuki_contr)
    # mayc_checkpt_dir = os.path.join("graph_checkpts", MAYC_contr)
    # meebits_checkpt_dir = os.path.join("graph_checkpts", meebits_contr)

    #############
    # construct graphs

    g = construct_graph(azuki_contr, offset=5000, resume_from_dir=azuki_checkpt_dir)

    #######
    # intersection of constructed graphs
    # g1 = nx.read_gml("Constructed_Graph_big.gml")
    # g2 = nx.read_gml("Meebits.gml")
    # bayc_meebits = nx.intersection(g1, g2)
    # nx.write_gml(bayc_meebits, "BAYC_Meebits_Int.gml")
    # print("Nodes: ", bayc_meebits.number_of_nodes())
    # print("Edges: ", bayc_meebits.number_of_edges())
    #######
    # compose graphs
    # g1 = nx.read_gml("Constructed_Graph_big.gml")
    # g2 = nx.read_gml("Meebits.gml")
    # g3 = nx.read_gml("Doodles.gml")
    # g4 = nx.read_gml("MAYC.gml")
    # g5 = nx.read_gml("Moonbird_Graph.gml")
    # g6 = nx.read_gml("Azuki.gml")
    # g7 = nx.read_gml("Terraforms.gml")
    # all_contracts = {}
    # constructed_graph_big.gml
    # all_contracts["BAYC"] = g1
    # all_contracts["Meebits"] = g2
    # all_contracts["Doodles"] = g3
    # all_contracts["MAYC"] = g4
    # all_contracts["Moonbirds"] = g5

    # g = nx.compose_all((g1, g2, g3, g4, g5, g6, g7))
    # g_w = nx.compose_all((g2, g7))
    # nx.write_gml(g, "Composed_Doodle_Moon_bayc_mayc.gml")
    #######
    # status report after graph is processed
    # print("Nodes: ", g.number_of_nodes())
    # print("Edges: ", g.number_of_edges())
    #######
    # save graph
    # g = nx.read_gml("Constructed_Graph.gml")
    # nx.write_gml(g, "Constructed_Graph_big.gml")
    # nx.write_gml(g, "Terraforms_Graph.gml")
    nx.write_gml(g, "Azuki.gml")
    # nx.write_gml(g, "All_Graphs(with Washed).gml")
    # nx.write_gml(g_w, "All_Washed.gml")
    #######
    #######
    # data analysis
    # undir_g = nx.Graph(g)
    # node1 = random.choice(list(g))
    # node2 = random.choice(list(g))
    # print("Shortest parths: ", nx.shortest_path(undir_g, node1, node2))
    # total_path = 0
    # for _ in range(10000):
    #     total_path += len(nx.shortest_path(undir_g, node1, node2))
    #     node1 = random.choice(list(g))
    #     node2 = random.choice(list(g))
    # print("Avg path length: ", total_path/10000)
    # print(
    #     "Common NEIGHBOURS: ", [n for n in nx.common_neighbors(undir_g, node1, node2)]
    # )
    ####### remove contracts using contract list csv
    # for name, graph in all_contracts.items():
    #     g = remove_contracts(
    #         graph,
    #         os.path.join(
    #             "eth_contracts",
    #             "export-verified-contractaddress-opensource-license.csv",
    #         ),
    #     )
    #     print("without contracts: ", len(g.nodes()))
    #     print("with contracts: ", len(graph.nodes()))
    #     nx.write_gml(g, name + "_no_contracts.gml")
    # print("0xC310e760778ECBca4C65B6C559874757A4c4Ece0".lower() in contr_addrs)


if __name__ == "__main__":
    _main()
