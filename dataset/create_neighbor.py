from collections import defaultdict
import time
import argparse
id2entity_name = defaultdict(str)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None)
args = parser.parse_args()

# dataset_name = 'FB15k-237'

with open('dataset/' + args.dataset + '/get_neighbor/entity2id.txt', 'r') as file:
    entity_lines = file.readlines()
    for line in entity_lines:
        _name, _id = line.strip().split("\t")
        id2entity_name[int(_id)] = _name

id2relation_name = defaultdict(str)

with open('dataset/' + args.dataset + '/get_neighbor/relation2id.txt', 'r') as file:
    relation_lines = file.readlines()
    for line in relation_lines:
        _name, _id = line.strip().split("\t")
        id2relation_name[int(_id)] = _name

train_triplet = []


for line in open('dataset/' + args.dataset + '/get_neighbor/train2id.txt', 'r'):
    head, relation, tail = line.strip('\n').split()
    train_triplet.append(list((int(head), int(relation), int(tail))))

for line in open('dataset/' + args.dataset + '/get_neighbor/test2id.txt', 'r'):
    head, relation, tail = line.strip('\n').split()
    train_triplet.append(list((int(head), int(relation), int(tail))))

for line in open('dataset/'+args.dataset+'/get_neighbor/valid2id.txt', 'r'):
    head, relation, tail = line.strip('\n').split()
    train_triplet.append(list((int(head), int(relation), int(tail))))


graph = {}
reverse_graph = {}

def init_graph(graph_triplet):

        for triple in graph_triplet:
            head = triple[0]
            rela = triple[1]
            tail = triple[2]

            if(head not in graph.keys()):
                graph[head] = {}
                graph[head][tail] = rela
            else:
                graph[head][tail] = rela

            if(tail not in reverse_graph.keys()):
                reverse_graph[tail] = {}
                reverse_graph[tail][head] = rela
            else:
                reverse_graph[tail][head] = rela
        
        # return graph, reverse_graph, node_indegree, node_outdegree

init_graph(train_triplet)



import random

def random_delete(triplet, reserved_num): 
    reserved = random.sample(triplet, reserved_num)
    return reserved

def get_onestep_neighbors(graph, source, sample_num):
    triplet = []
    try:
        nei = list(graph[source].keys())
        # nei = random.sample(graph[source].keys(), sample_num)
        triplet = [tuple((source, graph[source][nei[i]], nei[i])) for i in range(len(nei))]
    except KeyError:
        pass
    except ValueError:
        nei = list(graph[source].keys())
        triplet = [tuple((source, graph[source][nei[i]], nei[i])) for i in range(len(nei))]
    return triplet

def get_entity_neighbors(traget_entity, max_triplet):

    as_head_neighbors = get_onestep_neighbors(graph, traget_entity, max_triplet // 2)
    as_tail_neighbors = get_onestep_neighbors(reverse_graph, traget_entity, max_triplet // 2)

    all_triplet = as_head_neighbors + as_tail_neighbors

    return all_triplet

def get_triplet(triplet):
    head_entity = triplet[0]
    tail_entity = triplet[2]
    triplet = tuple((triplet[0], triplet[1], triplet[2]))

    head_triplet = get_entity_neighbors(head_entity, 4)
    tail_triplet = get_entity_neighbors(tail_entity, 4)

    temp_triplet = list(set(head_triplet + tail_triplet))
    temp_triplet = list(set(temp_triplet) - set([triplet]))
    # if len(temp_triplet) > 8:
    #     del_triplet = list(set(temp_triplet) - set([triplet]))
        # temp_triplet = random_delete(del_triplet, 7)

    return temp_triplet



import copy

def change_(triplet_list):
    tri_text = []
    for item in triplet_list:
        # text = id2entity_name[item[0]] + '\t' + id2relation_name[item[1]] + '\t' + id2entity_name[item[2]]
        h = id2entity_name[item[0]]
        r = id2relation_name[item[1]]
        t = id2entity_name[item[2]]
        tri_text.append([h, r, t])
    return tri_text

mask_idx = 99999999
masked_tail_neighbor = defaultdict(list)
masked_head_neighbor = defaultdict(list)
for triplet in train_triplet:
    tail_masked = copy.deepcopy(triplet)
    head_masked = copy.deepcopy(triplet)
    tail_masked[2] = mask_idx
    head_masked[0] = mask_idx
    masked_tail_neighbor['\t'.join([id2entity_name[triplet[0]], id2relation_name[triplet[1]]])] = change_(get_triplet(tail_masked))
    masked_head_neighbor['\t'.join([id2entity_name[triplet[2]], id2relation_name[triplet[1]]])] = change_(get_triplet(head_masked))


import json

with open("dataset/" + args.dataset + "/masked_tail_neighbor.txt", "w") as file:
    file.write(json.dumps(masked_tail_neighbor, indent=1))

with open("dataset/" + args.dataset + "/masked_head_neighbor.txt", "w") as file:
    file.write(json.dumps(masked_head_neighbor, indent=1))





