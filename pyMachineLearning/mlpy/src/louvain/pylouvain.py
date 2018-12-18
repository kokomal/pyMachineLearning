# coding:utf-8

'''
    Implements the Louvain method.
    Input: a weighted undirected graph
    Ouput: a (partition, modularity) pair where modularity is maximum
'''


class PyLouvain:

    '''
        Builds a graph from _path.
        _path: a path to a file containing "node_from node_to" edges (one per line)
    '''

    @classmethod
    def from_file(cls, path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        nodes = {}
        edges = []
        for line in lines:
            n = line.split()
            if not n:
                break
            nodes[n[0]] = 1
            nodes[n[1]] = 1
            w = 1
            if len(n) == 3:
                w = int(n[2])
            edges.append(((n[0], n[1]), w))
        print("="*70)
        print(nodes)
        print("="*70)
        print(edges)
        print("="*70)
        # rebuild graph with successive identifiers
        nodes_, edges_ = in_order(nodes, edges)
        print("%d nodes, %d edges" % (len(nodes_), len(edges_)))
        return cls(nodes_, edges_)

    '''
        Builds a graph from _path.
        _path: a path to a file following the Graph Modeling Language specification
    '''

    @classmethod
    def from_gml_file(cls, path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        nodes = {}
        edges = []
        current_edge = (-1, -1, 1)
        in_edge = 0
        for line in lines:
            words = line.split()
            if not words:
                break
            if words[0] == 'id':
                nodes[int(words[1])] = 1
            elif words[0] == 'source':
                in_edge = 1
                current_edge = (int(words[1]), current_edge[1], current_edge[2])
            elif words[0] == 'target' and in_edge:
                current_edge = (current_edge[0], int(words[1]), current_edge[2])
            elif words[0] == 'value' and in_edge:
                current_edge = (current_edge[0], current_edge[1], int(words[1]))
            elif words[0] == ']' and in_edge:
                edges.append(((current_edge[0], current_edge[1]), 1))
                current_edge = (-1, -1, 1)
                in_edge = 0
        nodes, edges = in_order(nodes, edges)
        print("%d nodes, %d edges" % (len(nodes), len(edges)))
        return cls(nodes, edges)

    '''
        Initializes the method.
        _nodes: a list of ints
        _edges: a list of ((int, int), weight) pairs
    '''

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        # precompute m (sum of the weights of all links in network)
        #            k_i (sum of the weights of the links incident to node i)
        self.m = 0
        self.k_i = [0 for n in nodes]
        self.edges_of_node = {}
        self.w = [0 for n in nodes]
        for e in edges:
            self.m += e[1]  # 每一条边的权重都叠加
            self.k_i[e[0][0]] += e[1]  # 边的两端都叠加计算，Ki为节点i的连接边所有权重之和
            self.k_i[e[0][1]] += e[1]  # 同上
            # save edges by node
            if e[0][0] not in self.edges_of_node:  # 不在里面则新建
                self.edges_of_node[e[0][0]] = [e]  # edges_of_node = {'node',[edge1, edge2...]}
            else:
                self.edges_of_node[e[0][0]].append(e)  # 在里面则append
            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)
        # access community of a node in O(1) time
        self.communities = [n for n in nodes]
        self.actual_partition = []

    '''
        Applies the Louvain method.
    '''

    def apply_method(self):
        network = (self.nodes, self.edges)
        #best_partition = [[node] for node in network[0]]
        best_q = -1
        i = 1
        while 1:
            print("pass #%d" % i)
            i += 1
            partition = self.first_phase(network)
            print("FIRST PARTITION IS ", partition)
            q = self.compute_modularity(partition)
            partition = [c for c in partition if c]  # 清洗空洞
            print("AFTER CLEANSE PARTITION IS ", partition)
            # print("%s (%.8f)" % (partition, q))
            # clustering initial nodes with partition
            if self.actual_partition:
                actual = []
                for p in partition:
                    part = []
                    for n in p:
                        part.extend(self.actual_partition[n])
                    actual.append(part)
                self.actual_partition = actual
            else:
                self.actual_partition = partition
            print("ACTUAL PARTITION IS ", self.actual_partition)
            if q == best_q:  # q无法再优化，则退出
                break
            network = self.second_phase(network, partition)  # 更新network
            #best_partition = partition
            best_q = q
        return (self.actual_partition, best_q)

    '''
        Computes the modularity of the current network.
        _partition: a list of lists of nodes
    '''

    def compute_modularity(self, partition):  # Q ec-αc平方
        q = 0
        m2 = self.m * 2
        for i in range(len(partition)):
            q += self.s_in[i] / m2 - (self.s_tot[i] / m2) ** 2
        return q

    '''
        Computes the modularity gain of having node in community _c.
        _node: an int
        _c: an int
        _k_i_in: the sum of the weights of the links from _node to nodes in _c
    '''

    def compute_modularity_gain(self, node, c, k_i_in):  # deltaQ
        return 2 * k_i_in - self.s_tot[c] * self.k_i[node] / self.m

    '''
        Performs the first phase of the method.
        _network: a (nodes, edges) pair
    '''

    def first_phase(self, network):
        # make initial partition
        best_partition = self.make_initial_partition(network)  # 初始化的best_partition为完整的nodelist[[]]
        while 1:
            improvement = 0
            for node in network[0]:
                node_community = self.communities[node]  # communities是一个[node]，node为序号，value为社区归属
                # default best community is its own
                best_community = node_community
                best_gain = 0
                # remove _node from its community
                best_partition[node_community].remove(node)  # 先将自己的community从best_partition中移走
                best_shared_links = 0
                for e in self.edges_of_node[node]:
                    if e[0][0] == e[0][1]:
                        continue
                    if e[0][0] == node and self.communities[e[0][1]] == node_community or e[0][1] == node and self.communities[e[0][0]] == node_community:
                        best_shared_links += e[1]
                self.s_in[node_community] -= 2 * (best_shared_links + self.w[node])  # 扣除此原先community的s_in, 2倍边权重 + 自身w权重（自己被除名，因此瓜葛要清算）
                self.s_tot[node_community] -= self.k_i[node]  # s_tot的扣除简单，即扣除k_i即可
                self.communities[node] = -1  # 待定，尚不清楚新的community是什么，暂记-1
                communities = {}  # only consider neighbors of different communities
                for neighbor in self.get_neighbors(node):  # 遍历node所有的neighbor，找到最大的gain
                    nb_community = self.communities[neighbor]
                    if nb_community in communities:  # 勿重复，因为neighbor可能很多且可能从属于同一个community
                        continue
                    communities[nb_community] = 1
                    shared_links = 0
                    for e in self.edges_of_node[node]:  # 算一遍node的边，注意此时nb_community为neighbor所在社区
                        if e[0][0] == e[0][1]:
                            continue
                        if e[0][0] == node and self.communities[e[0][1]] == nb_community or e[0][1] == node and self.communities[e[0][0]] == nb_community:
                            shared_links += e[1]
                    # compute modularity gain obtained by moving _node to the community of _neighbor
                    gain = self.compute_modularity_gain(node, nb_community, shared_links)  # 新人加入，2倍shared_links，gain少除了一次m，但无伤大雅
                    if gain > best_gain:
                        best_community = nb_community
                        best_gain = gain
                        best_shared_links = shared_links
                # insert _node into the community maximizing the modularity gain
                best_partition[best_community].append(node)  # 最优归属
                self.communities[node] = best_community  # 确定最终communities
                self.s_in[best_community] += 2 * (best_shared_links + self.w[node])
                self.s_tot[best_community] += self.k_i[node]
                if node_community != best_community:
                    improvement = 1
            if not improvement:
                break
        return best_partition

    '''
        Yields the nodes adjacent to _node.
        _node: an int
    '''

    def get_neighbors(self, node):
        for e in self.edges_of_node[node]:
            if e[0][0] == e[0][1]:  # a node is not neighbor with itself
                continue
            if e[0][0] == node:
                yield e[0][1]
            # if e[0][1] == node:
            else:  # 简易修复冗余代码
                yield e[0][0]

    '''
        Builds the initial partition from _network.
        _network: a (nodes, edges) pair
    '''

    # network不动，初始化了s_in,s_tot,返回[[node]]复合数组作为初始partition
    def make_initial_partition(self, network):  # network=(nodes, edges)
        partition = [[node] for node in network[0]]
        print("INITIALLY， THE PARTITION IS ", partition)
        self.s_in = [0 for node in network[0]]  # 初始化为[0...0]，这是因为初始化每一个community只有1个成员，因此内部的权重默认为0，再考虑自我指向的情况
        self.s_tot = [self.k_i[node] for node in network[0]]  # 初始化为ki
        for e in network[1]:
            if e[0][0] == e[0][1]:  # only self-loops 自循环，s_in要double
                self.s_in[e[0][0]] += 2 * e[1]
                # self.s_in[e[0][1]] += e[1]
        return partition

    '''
        Performs the second phase of the method.
        _network: a (nodes, edges) pair
        _partition: a list of lists of nodes
    '''

    def second_phase(self, network, partition):
        nodes_ = [i for i in range(len(partition))]  # 缩编的新的nodes，连续
        # print("SECOND PHASE ", nodes_)
        # relabelling communities
        communities_ = []
        d = {}
        i = 0
        print("OLD COMMUNITY " , self.communities)
        for community in self.communities:
            if community in d:
                communities_.append(d[community])
            else:
                d[community] = i
                communities_.append(i)
                i += 1
        self.communities = communities_
        print("NEW COMMUNITY " , self.communities)  # 归一化community，原先序号是按照原始的node节点贴标签的，现在要精简化
        # building relabelled edges
        edges_ = {}
        for e in network[1]:
            ci = self.communities[e[0][0]]  # 现在的ci和cj都是新的community归属的标签
            cj = self.communities[e[0][1]]
            try:
                edges_[(ci, cj)] += e[1]
            except KeyError:
                edges_[(ci, cj)] = e[1]
        edges_ = [(k, v) for k, v in edges_.items()]  # 这样一来nodes和edges已经重新算好了
        # recomputing k_i vector and storing edges by node
        self.k_i = [0 for n in nodes_]  # 接下来算权重和ki，还有各个节点的边
        self.edges_of_node = {}
        self.w = [0 for n in nodes_]  # 自有权重w复位
        for e in edges_:
            self.k_i[e[0][0]] += e[1]
            self.k_i[e[0][1]] += e[1]
            if e[0][0] == e[0][1]:  # 自己指向自己，则w要递增
                self.w[e[0][0]] += e[1]
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]] = [e]
            else:
                self.edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)
        # resetting communities
        self.communities = [n for n in nodes_]
        return (nodes_, edges_)


'''
    Rebuilds a graph with successive nodes' ids.
    _nodes: a dict of int
    _edges: a list of ((int, int), weight) pairs
'''


def in_order(nodes, edges):
        # rebuild graph with successive identifiers
        nodes = list(nodes.keys())
        nodes.sort()
        print("="*80)
        print(nodes)
        i = 0
        nodes_ = []
        d = {}
        for n in nodes:
            nodes_.append(i)
            d[n] = i
            i += 1
        edges_ = []
        for e in edges:
            edges_.append(((d[e[0][0]], d[e[0][1]]), e[1]))
        return (nodes_, edges_)
