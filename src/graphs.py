import torch
import math
from torch_geometric.data import Data
from transformers import DistilBertTokenizer, DistilBertModel

HEURISTIC_THRESHOLD = 50
EMBEDDING_DIM = 768

# Load the text embedding model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")


def get_text_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Get the mean embeddings
    return embeddings


def create_graph(elements, tokenizer=tokenizer, model=text_model):
    nodes = []
    edges = []
    edge_attrs = []

    texts = [element[1] for element in elements]
    embeddings = get_text_embeddings(texts, tokenizer, model)

    for i, (element, embedding) in enumerate(zip(elements, embeddings)):
        box, text = element
        nodes.append({"text": text, "bbox": box, "embedding": embedding})

    for i, elem_i in enumerate(nodes):
        for j, elem_j in enumerate(nodes):
            if i != j:
                if is_spatially_proximate(elem_i["bbox"], elem_j["bbox"]):
                    edges.append((i, j))
                    edge_attrs.append(get_edge_attributes(elem_i, elem_j))

    x = torch.stack([node["embedding"] for node in nodes])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
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
    horizontal_distance = (
        min(horizontal_distances) if left1 > right2 or left2 > right1 else 0
    )
    vertical_distance = (
        min(vertical_distances) if top1 > bottom2 or top2 > bottom1 else 0
    )

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
    combined_edge_attr = (
        torch.cat(combined_edge_attr, dim=0) if combined_edge_attr else None
    )

    merged_graph = Data(
        x=combined_x, edge_index=combined_edge_index, edge_attr=combined_edge_attr
    )
    return merged_graph, updated_nodes, updated_edges