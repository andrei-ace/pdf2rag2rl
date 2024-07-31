import torch
import math
import hashlib
from torch_geometric.data import Data

from embeddings import get_text_embeddings


HEURISTIC_THRESHOLD = 50

def create_graph(elements):
    nodes = []
    edges = []

    texts = [element[1] for element in elements]
    embeddings = get_text_embeddings(texts)

    for i, (element, embedding) in enumerate(zip(elements, embeddings)):
        box, text = element
        nodes.append({"text": text, "bbox": box, "embedding": embedding})

    for i, elem_i in enumerate(nodes):
        for j, elem_j in enumerate(nodes):
            if i != j:
                if is_spatially_proximate(elem_i["bbox"], elem_j["bbox"]):
                    edges.append((i, j))

    x = torch.stack([node["embedding"] for node in nodes])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data, nodes, edges


def min_distance_between_edges(box1, box2):
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2

    # Horizontal distance between the left and right edges
    horizontal_distances = [
        abs(left1 - right2),  # Distance from left edge of box1 to right edge of box2
        abs(right1 - left2),  # Distance from right edge of box1 to left edge of box2
    ]

    # Vertical distance between the top and bottom edges
    vertical_distances = [
        abs(top1 - bottom2),  # Distance from top edge of box1 to bottom edge of box2
        abs(bottom1 - top2),  # Distance from bottom edge of box1 to top edge of box2
    ]

    # Calculate distances only if boxes are not overlapping
    horizontal_distance = min(horizontal_distances) if left1 > right2 or left2 > right1 else 0
    vertical_distance = min(vertical_distances) if top1 > bottom2 or top2 > bottom1 else 0

    # Euclidean distance if both horizontal and vertical distances are non-zero
    if horizontal_distance and vertical_distance:
        return math.sqrt(horizontal_distance**2 + vertical_distance**2)
    # Otherwise, return the maximum of the distances (one of them will be zero)
    return max(horizontal_distance, vertical_distance)


def is_spatially_proximate(box1, box2, threshold=HEURISTIC_THRESHOLD):
    distance = min_distance_between_edges(box1, box2)
    return distance < threshold


def get_edge_attributes(elem_i, elem_j):
    box_i = elem_i["bbox"]
    box_j = elem_j["bbox"]
    return [abs(box_i[0] - box_j[0]), abs(box_i[1] - box_j[1])]


def update_coordinates_and_merge_graphs(graphs_nodes_edges, images):
    # Initialize empty lists for combined node features, edges, and edge attributes
    combined_x = []
    combined_edge_index = []
    combined_edge_attr = []
    updated_nodes = []
    updated_edges = []
    total_height = 0

    node_offset = 0
    for (graph, nodes, edges), img in zip(graphs_nodes_edges, images):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr

        # Update node coordinates and add to combined list
        for i, node in enumerate(nodes):
            bbox = node["bbox"]
            updated_bbox = [
                bbox[0],
                bbox[1] + total_height,
                bbox[2],
                bbox[3] + total_height,
            ]
            node["bbox"] = updated_bbox
            updated_nodes.append(node)

        # Add node features and edges to combined lists
        combined_x.append(x)
        combined_edge_index.append(edge_index + node_offset)
        if edge_attr is not None:
            combined_edge_attr.append(edge_attr)

        # Update edge indices to account for node offset
        for edge in edges:
            updated_edges.append((edge[0] + node_offset, edge[1] + node_offset))

        node_offset += x.size(0)
        total_height += img.size[1]

    # Concatenate all the node features, edges, and edge attributes
    combined_x = torch.cat(combined_x, dim=0)
    combined_edge_index = torch.cat(combined_edge_index, dim=1)
    combined_edge_attr = torch.cat(combined_edge_attr, dim=0) if combined_edge_attr else None

    merged_graph = Data(x=combined_x, edge_index=combined_edge_index, edge_attr=combined_edge_attr)
    return merged_graph, updated_nodes, updated_edges


def tensor_to_tuple(tensor):
    if isinstance(tensor, torch.Tensor):
        return tuple(tensor.tolist())
    return tensor


def compute_graph_hash(graph, nodes, edges):
    # Convert tensor attributes in nodes to tuples for consistent sorting
    sorted_nodes = sorted(nodes, key=lambda x: tuple((k, tensor_to_tuple(v)) for k, v in sorted(x.items())))

    # Convert edges to tuples if they are tensors
    sorted_edges = sorted((tuple(edge) if isinstance(edge, torch.Tensor) else edge) for edge in edges)

    # Create a string representation of the sorted nodes and edges
    nodes_str = "".join([str(node) for node in sorted_nodes])
    edges_str = "".join([str(edge) for edge in sorted_edges])

    # Combine the graph features, sorted nodes, and sorted edges into a single string
    combined_str = str(graph.x.tolist()) + nodes_str + edges_str

    # Compute the hash value using SHA-256
    hash_value = hashlib.sha256(combined_str.encode()).hexdigest()

    return hash_value


def find_connected_components(edge_index, num_nodes):
    parent = list(range(num_nodes))

    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(v1, v2):
        root1 = find(v1)
        root2 = find(v2)
        if root1 != root2:
            parent[root2] = root1

    for i in range(edge_index.size(1)):
        union(edge_index[0, i].item(), edge_index[1, i].item())

    components = {}
    for node in range(num_nodes):
        root = find(node)
        if root not in components:
            components[root] = []
        components[root].append(node)

    return list(components.values())


def split_graph(graph, nodes, edges):
    # Find connected components in the graph
    components = find_connected_components(graph.edge_index, graph.num_nodes)

    subgraphs = []
    for component in components:
        # Get node indices and edges for the subgraph
        subgraph_nodes = component
        subgraph_node_idx = {node: i for i, node in enumerate(subgraph_nodes)}
        subgraph_edges = [
            (subgraph_node_idx[u], subgraph_node_idx[v])
            for u, v in edges
            if u in subgraph_node_idx and v in subgraph_node_idx
        ]

        # Create the subgraph Data object
        subgraph_x = graph.x[subgraph_nodes]
        subgraph_edge_index = torch.tensor(subgraph_edges, dtype=torch.long).t().contiguous()
        subgraph = Data(x=subgraph_x, edge_index=subgraph_edge_index)

        # Create the subgraph's nodes and edges list
        subgraph_nodes_list = [nodes[node] for node in subgraph_nodes]
        subgraph_edges_list = subgraph_edges

        subgraphs.append((subgraph, subgraph_nodes_list, subgraph_edges_list))

    return subgraphs

def extract_text_from_graph(graph, nodes, edges):
    node_texts = [node["text"] for node in nodes if "text" in node]
    # Concatenate all text into a single string
    concatenated_text = " ".join(node_texts)
    return concatenated_text