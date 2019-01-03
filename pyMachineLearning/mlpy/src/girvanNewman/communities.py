# coding:utf-8
"""
Author: Lingzhe Teng
Date: Nov. 15, 2015

"""
from sqlalchemy.dialects.postgresql.hstore import idx_precedence

"""
Executing code: 
Python communities.py input.txt image.png

"""

"""
Change log:

- Nov. 26
1. saving image based on the input image name instead of show graph

- Nov. 28
1. fix bug for ploting community with random colors

"""

from operator import mul
import networkx as nx
import os
import sys
# import community
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mpcolors
import matplotlib.cm as mpcm
import numpy as np

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                            3rd-party Libraries                               """
"""                                                                              """    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
## networkx(betweenness)
Home: https://networkx.github.io

## community(modularity)
Home: http://perso.crans.org/aynaud/communities/
API: http://perso.crans.org/aynaud/communities/api.html

## matplotlib
Home: http://matplotlib.org/index.html
Doc: http://matplotlib.org/examples/index.html
Ref: https://gist.github.com/shobhit/3236373

"""

eps = 1e-7
class Communities:

    def __init__(self, ipt_txt, ipt_png):
        self.ipt_txt = ipt_txt
        self.ipt_png = ipt_png
        self.graph = None

    def initialize(self):
        if not os.path.isfile(self.ipt_txt):
            self.quit(self.ipt_txt + " doesn't exist or it's not a file")

        # initialize 3rd-party libraries
        self.graph = nx.Graph()
        # load data
        self.load_txt(self.ipt_txt)
        
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                               Main Functions                                 """
    """                                                                              """    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def find_best_partition(self):
        from community import community_louvain
        G = self.graph.copy()
        modularity = -float('inf') 
        removed_edges = []
        partition = {}
        cou = 0
        while cou < 40:
            cou += 1
            betweenness = self.calculte_betweenness(G)  # 1.算介度
            max_betweenness_edges = self.get_max_betweenness_edges(betweenness)  # 2.根据介度算最大介度的边的集合
            if len(G.edges()) == len(max_betweenness_edges): # 介度全部都一样，退出
                break

            G.remove_edges_from(max_betweenness_edges)  # 将最大介度的边全部移除
            components = nx.connected_components(G) # 获得连通的所有components
            idx = 0
            tmp_partition = {}
            for component in components:
                for inner in list(component):
                    tmp_partition.setdefault(inner, idx) # 先获得暂时的分区，按顺序递增
                idx += 1
                print('IDX=', idx)
            cur_mod = community_louvain.modularity(tmp_partition, G) # 调用louvain的模块算模块度
            print("CUR MOD=", cur_mod, 'while modularity=', modularity)
            if cur_mod < modularity:# or abs(cur_mod - modularity) < eps: # 模块度小了，说明不能再划分，则此次分割无效，要补回去，并且退出
                G.add_edges_from(max_betweenness_edges)
                break;
            else:
                partition = tmp_partition
            #partition = tmp_partition
            removed_edges.extend(max_betweenness_edges) # 删掉的最大介度的边集合，不断更新
            modularity = cur_mod
            self.display(partition)
        print('COUNT', cou)
        return partition, G, removed_edges

    def get_max_betweenness_edges(self, betweenness):
        print('btw', betweenness)
        max_betweenness_edges = []
        max_betweenness = max(betweenness.items(), key=lambda x: x[1])
        print('MB-',max_betweenness)
        for (k, v) in betweenness.items():
            if v == max_betweenness[1]:
                max_betweenness_edges.append(k)
        print('MBE--', max_betweenness_edges)
        return max_betweenness_edges

    def calculte_betweenness(self, G, bonus=True):
        """
        Calculate Betweenness
        input:
        - G: graph
        - bonus: True if use my own betweenness calculator. (bonus=True by default)

        """
        if bonus:
            betweenness = self.my_betweenness_calculation(G)
        else:
            betweenness = nx.edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None)
        return betweenness

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                               Bonus Functions                                """
    """                                                                              """    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def build_level(self, G, root):
        """ 
        Build level for graph

        input:
        - G: networkx graph
        - root: root node

        output:
        - levels: nodes in each level
        - predecessors: predecessors for each node
        - successors: successors for each node
 
        """
        levels = {}
        predecessors = {}
        successors = {}

        cur_level_nodes = [root]  # initialize start point
        nodes = []  # store nodes that have been accessed
        level_idx = 0  # track level index
        while cur_level_nodes:  # if have nodes for a level, continue process
            nodes.extend(cur_level_nodes)  # add nodes that are inside new level into nodes list
            levels.setdefault(level_idx, cur_level_nodes)  # set nodes for current level
            next_level_nodes = []  # prepare nodes for next level

            # find node in next level
            for node in cur_level_nodes:  # 初始化为root的所有neighbor
                nei_nodes = G.neighbors(node)  # all neighbors for the node in current level
                
                # find neighbor nodes in the next level
                for nei_node in nei_nodes:
                    if nei_node not in nodes:  # nodes in the next level must not be accessed
                        predecessors.setdefault(nei_node, [])  # initialize predecessors dictionary, use a list to store all predecessors
                        predecessors[nei_node].append(node) 
                        successors.setdefault(node, [])  # initialize successors dictionary, use a list to store all successors
                        successors[node].append(nei_node)

                        if nei_node not in next_level_nodes:  # avoid add same node twice
                            next_level_nodes.append(nei_node)
            cur_level_nodes = next_level_nodes
            level_idx += 1
        return levels, predecessors, successors

    def calculate_credits(self, G, levels, predecessors, successors, nodes_nsp):
        """
        Calculate credits for nodes and edges

        """
        nodes_credit = {}
        edges_credit = {}

        # loop, from bottom to top, not including the zero level
        for lvl_idx in range(len(levels) - 1, 0, -1): # 从底向上遍历，不包括root
            lvl_nodes = levels[lvl_idx]  # get nodes in the level 获得每一层的节点集合

            # calculate for each node in current level
            for lvl_node in lvl_nodes: # 在这一层，遍历其所有的节点
                nodes_credit.setdefault(lvl_node, 1.)  # set default credit for the node, 1
                if lvl_node in successors.keys():  # if it is not a leaf node 叶节点不可能有successor
                    # Each node that is not a leaf gets credit = 1 + sum of credits of the DAG edges from that node to level below
                    for successor in successors[lvl_node]:
                        nodes_credit[lvl_node] += edges_credit[(successor, lvl_node)]

                node_predecessors = predecessors[lvl_node]  #  get predecessors of the node in current level
                total_nodes_nsp = .0  # total number of shortest paths for predecessors of the node in current level
                
                # sum up for total_nodes_nsp
                for predecessor in node_predecessors:
                    total_nodes_nsp += nodes_nsp[predecessor]

                # again, calculate for the weight of each predecessor, and assign credit for the responding edge
                for predecessor in node_predecessors:
                    predecessor_weight = nodes_nsp[predecessor] / total_nodes_nsp  # calculate weight of predecssor 两次循环仅为算nsp的占比
                    edges_credit.setdefault((lvl_node, predecessor), nodes_credit[lvl_node] * predecessor_weight)  # bottom-up edge
        return nodes_credit, edges_credit

    def my_betweenness_calculation(self, G, normalized=True):
        """
        Main Bonus Function to calculation betweenness

        """
        graph_nodes = G.nodes()
        edge_contributions = {}
        rw = nx.connected_components(G)  # 获得连通图的list
        components = list(rw)  # connected components for current graph
        # calculate for each node
        for node in graph_nodes:
            # print('NODE', node)
            component = None  # the community current node belongs to
            for com in components: 
                if node in list(com):
                    component = list(com)  # 找到了该节点对应的社区，出于效率应该break[本人备注]
                    break
            nodes_nsp = {}  # number of shorest paths
            node_levels, predecessors, successors = self.build_level(G, node)  # build levels for calculation
            #print('==LEVELS==', node_levels, '==PRED==', predecessors, '==SUCC==', successors)
            # calculate shortest paths for each node (including current node)
            for each_node in component:
                shortest_paths = nx.all_shortest_paths(G, source=node, target=each_node)  # dijkstra法获得node出发的所有最短路径
                nodes_nsp[each_node] = len(list(shortest_paths))  # 获得各个节点的介数

            # calculate credits for nodes and edges (Only use "edges_credit" actually)
            nodes_credit, edges_credit = self.calculate_credits(G, node_levels, predecessors, successors, nodes_nsp)
            # print('EDGE-CREDITS', edges_credit)
            # sort tuple (key value of edges_credit), and sum up for edge_contributions
            for (k, v) in edges_credit.items():
                k = sorted(k, reverse=False)
                edge_contributions_key = (k[0], k[1])
                edge_contributions.setdefault(edge_contributions_key, 0)
                edge_contributions[edge_contributions_key] += v
           
        # divide by 2 to get true betweenness
        for (k, v) in edge_contributions.items():
            edge_contributions[k] = v / 2

        # normalize
        if normalized:
            max_edge_contribution = max(edge_contributions.values())
            for (k, v) in edge_contributions.items():
                edge_contributions[k] = v / max_edge_contribution
        # print('EDGE-CONTRIB', edge_contributions)
        return edge_contributions

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                               Plot Method                                    """
    """                                                                              """    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def plot_graph(self, part_graph, removed_edges):
        from functools import reduce
        G = self.graph
        G_part = part_graph
        exist_edges = part_graph.edges(data=True)
        pos = nx.spring_layout(G, k=0.1, iterations=50, scale=1.3)
        
        co = {1:'r', 2:'b', 3:'g', 4:'cyan', 5:'purple',6:'orange',7:'yellow',8:'darkgreen'}
        idx = 1
        # nodes
        coms = nx.connected_components(part_graph)
        for com in coms:
            nodes = list(com) 
            print("NODES are ", nodes)
            #np.random.seed(len(nodes) * sum(nodes) * reduce(mul, nodes, 1) * min(nodes) * max(nodes))
            colors = np.random.rand(4 if len(nodes) < 4 else len(nodes))
            colors = co[1 + idx % len(co)]
            idx+=1
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=500, node_color=colors)

        # edges
        nx.draw_networkx_edges(G, pos, edgelist=exist_edges, width=2, alpha=1, edge_color='k')
        nx.draw_networkx_edges(G, pos, edgelist=removed_edges, width=2, edge_color='k', style='dashed')

        # labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

        plt.axis('off')
        plt.savefig(self.ipt_png)
        # plt.show()

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                               Help Method                                    """
    """                                                                              """    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def load_txt(self, ipt_txt):        
        input_data = open(ipt_txt, 'rU')
        for line in input_data:
            line = line.strip('\n')
            line = str(line)
            line = line.split(" ")

            if len(line) != 2:
                self.quit("edge format for input.txt is error")

            ending_node_a = int(line[0])
            ending_node_b = int(line[1])
            self.graph.add_edge(ending_node_a, ending_node_b, weight=2.0, len=3.0)

    def display_graph(self):
        G = self.graph
        print ("+-------------------------------------------------+")
        print ("|                  Display Graph                  |")
        print ("+-------------------------------------------------+")

        print ("Number of Nodes:", G.number_of_nodes())
        print ("Number of Edges:", G.number_of_edges())
        print ("Nodes: \n", G.nodes())
        print ("Edges: \n", G.edges())

    def partition_to_community(self, partition):
        result = {}
        print("PARTITION", partition)
        for (k, v) in partition.items():
            result.setdefault(v, [])
            result[v].append(k)
        return result.values()

    def display(self, partition):
        comms = self.partition_to_community(partition)
        print('COMMS', comms)
        for comm in comms:
            comm.sort()  # actually duplicate process here
            print (comm)

    def quit(self, err_desc):
        raise SystemExit('\n' + "PROGRAM EXIT: " + err_desc + ', please check your input' + '\n')


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                               Main Method                                    """
"""                                                                              """    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':

    ipt_txt = "input.txt"
    ipt_png = "image.png"

    c = Communities(ipt_txt, ipt_png)
    c.initialize()
    c.display_graph()
    print("*"*70)
    partition, part_graph, removed_edges = c.find_best_partition()
    c.display(partition)
    c.plot_graph(part_graph, removed_edges)

