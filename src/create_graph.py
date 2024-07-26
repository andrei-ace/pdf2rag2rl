import torch
from torch_geometric.data import Data

def create_graph(elements):
    nodes = []
    edges = []
    edge_attrs = []

    for i, element in enumerate(elements):
        box, text = element
        nodes.append({
            'text': text,
            'bbox': box
        })

    for i, elem_i in enumerate(elements):
        for j, elem_j in enumerate(elements):
            if i != j:
                if is_spatially_proximate(elem_i[0], elem_j[0]):
                    edges.append((i, j))
                    edge_attrs.append(get_edge_attributes(elem_i, elem_j))

    x = torch.tensor([node['bbox'] for node in nodes], dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def is_spatially_proximate(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

def get_edge_attributes(elem_i, elem_j):
    box_i, _ = elem_i
    box_j, _ = elem_j
    return [abs(box_i[0] - box_j[0]), abs(box_i[1] - box_j[1])]