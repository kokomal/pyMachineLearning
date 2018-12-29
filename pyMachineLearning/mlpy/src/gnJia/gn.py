# import module
import re
import sys
from collections import Counter

# def func
def top_down(input,root):
    dict1 = dict()

    #start node
    dict1[root] = (0,1)

    # start layer
    cur_layer = [root]
    layer_index = 0
    edge = {}
    while cur_layer:
        layer_index += 1
        next_layer = []
        for parent_node in cur_layer:
            temp = []
            for i in range(len(data)):
                for ele in input:
                    if parent_node in ele:
                        for son_node in ele:
                            if son_node != parent_node and son_node not in cur_layer:
                                next_layer.append(son_node)
                                temp.append(son_node)
                        input.remove(ele)
            edge[parent_node] = temp

        for son_node in next_layer:
            if son_node in dict1:
                dict1[son_node] = (layer_index, dict1[son_node][1]+1)
            else:
                dict1[son_node] = (layer_index,1)
        cur_layer = next_layer
    return edge, dict1,layer_index-1


def bottom_up(edges, dict, layerIndex):
    score = {}
    edge_score = {}
    while layerIndex >= 0:
        for key, value in dict.items():
            if value[0] == layerIndex:
                score[key] = 1
                for i in edges[key]:
                    score[key] += score[i] * dict[key][1] / dict[i][1]
                    edge_score[tuple(sorted((key, i)))] = score[i]
        layerIndex -= 1
    return edge_score



# main func

if __name__ == '__main__':

    data = []

    with open(sys.argv[1]) as input:
        data = [re.findall(pattern="\w", string=line) for line in input]

    itemset = set([x for y in data for x in y])
    temp_score = {}
    test = []
    for start_root in itemset:
        with open(sys.argv[1]) as input:
            data = [re.findall(pattern="\w", string=line) for line in input]
        edges, dict1, layerIndex = top_down(data, start_root)
        edgeScore = bottom_up(edges, dict1, layerIndex)
        temp_score = Counter(temp_score) + Counter(edgeScore)
        # divided by 2
        final_score = sorted([(key, float(value) / 2) for key, value in temp_score.items()])

    with open(sys.argv[2], 'w') as file:
        for x in final_score:
            file.write(str(x)[1:-1].replace("'","")+ '\n')