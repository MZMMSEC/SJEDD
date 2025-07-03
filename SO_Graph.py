import torch, copy
import numpy as np
import itertools as it


# utils function
def bin_list(nsize):
    s = list(it.product(range(2), repeat=nsize))
    out = torch.tensor(s)
    return out


# =========# build graph for FFpp #===========================================

def legal_state_ffso(Ee, Eh, all_state, parent_dict, children_dict):
    state = copy.deepcopy(all_state)
    state = state.numpy()
    for edge in Ee:
        num = 1
        idx = []
        for i in range(len(state)):
            if state[i, edge[0]] == 1 and state[i, edge[1]] == 1:
               idx.append(i)
               num += 1
        state = np.delete(state, idx, axis=0)

    for edge in Eh:
        num = 1
        idx = []
        for i in range(len(state)):
            # leaf node and its state
            leaf_node_label = edge[1]
            leaf_state = state[i, edge[1]]
            if leaf_node_label <= 3:
                if state[i, edge[0]] == 0 and state[i, edge[1]] == 1:
                    idx.append(i)
                    num += 1

                #
                else:
                    child_nodes = children_dict[str(leaf_node_label)]
                    child_nodes_state = []
                    for k in range(len(child_nodes)):
                        child_nodes_state.append(state[i, child_nodes[k]])
                    if sum(child_nodes_state) == 0 and leaf_state == 1:
                        idx.append(i)
                        num += 1

            else:
                parent_nodes = parent_dict[str(leaf_node_label)]
                parent_nodes_state = []
                for j in range(len(parent_nodes)):
                    parent_nodes_state.append(state[i, parent_nodes[j]])
                if sum(parent_nodes_state) == 0 and leaf_state == 1:
                    idx.append(i)
                    num += 1
        state = np.delete(state, idx, axis=0)
    return torch.tensor(state)

def graph_SA_ffso():
    Eh_edge = [
        (0,1), (0,2), (0,3),
        (1, 6), (1, 7),
        (2, 4), (2, 5), (2, 6), (2, 7), (2,8), (2,9),
        (3, 4), (3, 6), (3, 7), (3, 8)
    ]
    Ee_edge = []
    # prepare a dict/list that record the parent nodes for each node
    parent_dict = {
        "4": [2, 3],
        "5": [2],
        "6": [1, 2, 3],
        "7": [1, 2, 3],
        "8": [2, 3],
        "9": [2],
    }
    children_dict = {
        "1": [6, 7],
        "2": [4, 5, 6, 7, 8, 9],
        "3": [4, 6, 7, 8],
    }

    state = legal_state_ffso(Ee_edge, Eh_edge, bin_list(10), parent_dict,
                        children_dict)  # all the legal states under the proposed graph

    label_state = get_label_state_idx_ffso(state)  # correspond to the leaf node w.r.t the graph state

    return state, label_state

def get_label_state_idx_ffso(state):
    length = state.shape[0]
    labels_id = {'1010111111': [], '1001000010': [], '1001001100': [], '1100001100': [],
                 '1001100000': [], '0000000000': []}
    for i in range(length):
        if (torch.tensor(label_map_ffso['1010111111']) == state[i]).all():
            labels_id['1010111111'] = i

        elif (torch.tensor(label_map_ffso['1001000010']) == state[i]).all():
            labels_id['1001000010'] = i

        elif (torch.tensor(label_map_ffso['1001001100']) == state[i]).all():
            labels_id['1001001100'] = i

        elif (torch.tensor(label_map_ffso['1100001100']) == state[i]).all():
            labels_id['1100001100'] = i

        elif (torch.tensor(label_map_ffso['1001100000']) == state[i]).all():
            labels_id['1001100000'] = i

        elif (torch.tensor(label_map_ffso['0000000000']) == state[i]).all():
            labels_id['0000000000'] = i

    return labels_id

label_map_ffso = {
    '1010111111': [1, 0,1,0, 1,1,1,1,1,1], # id
    '1001000010': [1, 0,0,1, 0,0,0,0,1,0], # phy_nose
    '1001001100': [1, 0,0,1, 0,0,1,1,0,0], # phy_mouth
    '1100001100': [1, 1,0,0, 0,0,1,1,0,0], # expr_mouth
    '1001100000': [1, 0,0,1, 1,0,0,0,0,0], # phys_eyes
    '0000000000': [0, 0,0,0, 0,0,0,0,0,0], # real
}


# =========# build graph for FFSC #===========================================
# define Graph and rules
def graph_SO_FFSC():
    Eh_edge = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 6), (1, 11),
        (2, 6), (2, 8), (2, 9), (2, 10), (2, 11),
        (3, 6), (3, 7), (3, 8), (3, 9), (3, 11),
        (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11),
        (5, 6), (5, 7), (5, 9), (5, 9), (5, 10), (5, 11),
    ]
    Ee_edge = []
    # prepare a dict/list that record the parent nodes for each node
    parent_dict = {
        "6": [1, 2, 3, 4, 5],
        "7": [3, 4, 5],
        "8": [2, 3, 4, 5],
        "9": [2, 3, 4, 5],
        "10": [2, 4, 5],
        "11": [1, 2, 3, 4, 5]
    }
    children_dict = {
        "1": [6, 11],
        "2": [6, 8, 9, 10, 11],
        "3": [6,7,8, 9, 11],
        "4": [6,7,8,9, 10, 11],
        "5": [6,7,8,9, 10, 11],
    }

    state = legal_state_ffsc(Ee_edge, Eh_edge, bin_list(12), parent_dict,
                        children_dict)  # all the legal states under the proposed graph

    label_state = get_label_state_idx_ffsc(state)  # correspond to the leaf node w.r.t the graph state

    return state, label_state

def legal_state_ffsc(Ee, Eh, all_state, parent_dict, children_dict):
    state = copy.deepcopy(all_state)
    state = state.numpy()
    for edge in Ee:
        num = 1
        idx = []
        for i in range(len(state)):
            if state[i, edge[0]] == 1 and state[i, edge[1]] == 1:
               idx.append(i)
               num += 1
        state = np.delete(state, idx, axis=0)

    for edge in Eh:
        num = 1
        idx = []
        for i in range(len(state)):
            # leaf node and its state
            leaf_node_label = edge[1]
            leaf_state = state[i, edge[1]]
            if leaf_node_label <= 5:
                if state[i, edge[0]] == 0 and state[i, edge[1]] == 1:
                    idx.append(i)
                    num += 1

                #
                else:
                    child_nodes = children_dict[str(leaf_node_label)]
                    child_nodes_state = []
                    for k in range(len(child_nodes)):
                        child_nodes_state.append(state[i, child_nodes[k]])
                    if sum(child_nodes_state) == 0 and leaf_state == 1:
                        idx.append(i)
                        num += 1

            else:
                parent_nodes = parent_dict[str(leaf_node_label)]
                parent_nodes_state = []
                for j in range(len(parent_nodes)):
                    parent_nodes_state.append(state[i, parent_nodes[j]])
                if sum(parent_nodes_state) == 0 and leaf_state == 1:
                    idx.append(i)
                    num += 1
        state = np.delete(state, idx, axis=0)
    return torch.tensor(state)


label_map = {
    '110000100001': [1,1,0,0,0,0,1,0,0,0,0,1], # age
    '101000001100': [1,0,1,0,0,0,0,0,1,1,0,0], # expr-smile
    '101000101111': [1,0,1,0,0,0,1,0,1,1,1,1], # expr-surprised
    '120110111101': [1,2,0,1,1,0,1,1,1,1,0,1], # gender
    '120010111111': [1,2,0,0,1,0,1,1,1,1,1,1], # ID
    '100001111111': [1,0,0,0,0,1,1,1,1,1,1,1], # pose
    '000000000000': [0,0,0,0,0,0,0,0,0,0,0,0], # real
}
def get_label_state_idx_ffsc(state):
    length = state.shape[0]
    labels_id = {'110000100001': [], '101000001100': [], '101000101111': [], '120110111101': [],
                 '120010111111': [], '100001111111': [], '000000000000': []}
    for i in range(length):
        if (torch.tensor(label_map['110000100001']) == state[i]).all():
            labels_id['110000100001'] = i

        elif (torch.tensor(label_map['101000001100']) == state[i]).all():
            labels_id['101000001100'] = i

        elif (torch.tensor(label_map['101000101111']) == state[i]).all():
            labels_id['101000101111'] = i

        elif (torch.tensor(label_map['120110111101']) == state[i]).all():
            labels_id['120110111101'] = i

        elif (torch.tensor(label_map['120010111111']) == state[i]).all():
            labels_id['120010111111'] = i

        elif (torch.tensor(label_map['100001111111']) == state[i]).all():
            labels_id['100001111111'] = i

        elif (torch.tensor(label_map['000000000000']) == state[i]).all():
            labels_id['000000000000'] = i
    return labels_id
