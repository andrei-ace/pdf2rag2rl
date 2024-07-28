from convert2image import convert_pdf_to_images
from detect_layout import detect_layout_elements
from models import AddNetwork, CriticNetwork, PolicyNetwork, RemoveNetwork, StopNetwork
from ocr import ocr_elements
from create_graph import create_graph, EMBEDDING_DIM
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
import cv2
from PIL import Image


def visualize_graph(image_pil, nodes, edges):
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Draw bounding boxes and node indices
    for idx, node in enumerate(nodes):
        box = node["bbox"]
        # Draw rectangle
        cv2.rectangle(image_cv, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
        # Calculate the position for the text (upper right corner)
        text_position = (box[2], box[1] + 0)  # Adjust the offset as needed
        # Put text at the calculated position
        cv2.putText(image_cv, str(idx), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # Draw edges
    for edge in edges:
        box_i = nodes[edge[0]]["bbox"]
        box_j = nodes[edge[1]]["bbox"]
        center_i = ((box_i[0] + box_i[2]) // 2, (box_i[1] + box_i[3]) // 2)
        center_j = ((box_j[0] + box_j[2]) // 2, (box_j[1] + box_j[3]) // 2)
        cv2.line(image_cv, center_i, center_j, color=(255, 0, 0), thickness=2)

    # Convert OpenCV image back to PIL format
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    # Display the image using matplotlib
    plt.figure(figsize=(12, 12))
    plt.imshow(image_pil)
    plt.axis("off")
    plt.show()


def get_possible_node_pairs(x, edge_index):
    node_count = x.size(0)
    all_pairs = [(i, j) for i in range(node_count) for j in range(node_count) if i != j]
    existing_edges = set(tuple(edge) for edge in edge_index.t().tolist())
    possible_add_pairs = [pair for pair in all_pairs if pair not in existing_edges]
    possible_remove_pairs = [pair for pair in all_pairs if pair in existing_edges]
    return possible_add_pairs, possible_remove_pairs


# Now you have graphs for each page in the PDF with text embeddings on the nodes
# and you can visualize one graph
def sample_action(graph, add_net, remove_net, critic_net, action_probs):

    # Sample an action based on the policy network output
    actions = ["add", "remove", "stop"]
    action_probs = action_probs.squeeze().detach().numpy()
    chosen_action = np.random.choice(actions, p=action_probs)
    print(f"Chosen action: {chosen_action}")

    # Create lists of possible node pairs for add and remove action# Create lists of possible node pairs for add and remove actions
    possible_add_pairs, possible_remove_pairs = get_possible_node_pairs(
        graph.x, graph.edge_index
    )

    if chosen_action == "add":
        # Calculate add probabilities for possible node pairs
        add_probs = []
        for pair in possible_add_pairs:
            node1_emb = graph.x[pair[0]].unsqueeze(0)
            node2_emb = graph.x[pair[1]].unsqueeze(0)
            add_prob = add_net(node1_emb, node2_emb)
            add_probs.append(add_prob.item())
        add_probs = torch.tensor(add_probs)
        add_probs = F.softmax(add_probs, dim=0)

        # Sample a specific add action
        add_action_idx = np.random.choice(
            len(possible_add_pairs), p=add_probs.detach().numpy()
        )
        add_action = possible_add_pairs[add_action_idx]
        print(f"Add edge between nodes: {add_action}")

        # Compute and print critic value
        node1_emb = graph.x[add_action[0]].unsqueeze(0)
        node2_emb = graph.x[add_action[1]].unsqueeze(0)
        critic_value = critic_net(
            graph.x,
            graph.edge_index,
            torch.tensor([1, 0, 0]).unsqueeze(0),
            node1_emb,
            node2_emb,
            torch.tensor([action_probs[0]]).unsqueeze(0),
        )
        print(f"Critic value: {critic_value.item()}")

    elif chosen_action == "remove":
        # Calculate remove probabilities for possible node pairs
        remove_probs = []
        for pair in possible_remove_pairs:
            node1_emb = graph.x[pair[0]].unsqueeze(0)
            node2_emb = graph.x[pair[1]].unsqueeze(0)
            remove_prob = remove_net(node1_emb, node2_emb)
            remove_probs.append(remove_prob.item())
        remove_probs = torch.tensor(remove_probs)
        remove_probs = F.softmax(remove_probs, dim=0)

        # Sample a specific remove action
        remove_action_idx = np.random.choice(
            len(possible_remove_pairs), p=remove_probs.detach().numpy()
        )
        remove_action = possible_remove_pairs[remove_action_idx]
        print(f"Remove edge between nodes: {remove_action}")

        # Compute and print critic value
        node1_emb = graph.x[remove_action[0]].unsqueeze(0)
        node2_emb = graph.x[remove_action[1]].unsqueeze(0)
        critic_value = critic_net(
            graph.x,
            graph.edge_index,
            torch.tensor([0, 1, 0]).unsqueeze(0),
            node1_emb,
            node2_emb,
            torch.tensor([action_probs[1]]).unsqueeze(0),
        )
        print(f"Critic value: {critic_value.item()}")

    else:
        print("Stop modifying the graph")

        # Compute and print critic value
        critic_value = critic_net(
            graph.x,
            graph.edge_index,
            torch.tensor([0, 0, 1]).unsqueeze(0),
            graph.x.mean(dim=0).unsqueeze(0),
            graph.x.mean(dim=0).unsqueeze(0),
            torch.tensor([action_probs[2]]).unsqueeze(0),
        )
        print(f"Critic value: {critic_value.item()}")


pdf_path = "docs/examples/1706.03762v7.pdf"
images = convert_pdf_to_images(pdf_path)

images = images[:1]  # Process only the first page for now

layout_boxes = [detect_layout_elements(image) for image in images]
ocr_results = [
    ocr_elements(image, boxes) for image, boxes in zip(images, layout_boxes)
]
graphs_and_nodes_edges = [create_graph(elements) for elements in ocr_results]

for (graph, nodes, edges), image in zip(graphs_and_nodes_edges, images):
    input_dim = EMBEDDING_DIM
    hidden_dim = 64
    projection_dim = 32

    # Initialize networks
    add_net = AddNetwork(input_dim, hidden_dim)
    remove_net = RemoveNetwork(input_dim, hidden_dim)
    stop_net = StopNetwork(input_dim, hidden_dim)
    policy_net = PolicyNetwork(input_dim, hidden_dim)
    critic_net = CriticNetwork(input_dim, hidden_dim, projection_dim)
    
    # Forward pass for policy network (example with graph state only)
    action_probs = policy_net(graph.x, graph.edge_index)
    sample_action(graph, add_net, remove_net, critic_net, action_probs)
    
    # Visualize the graph
    visualize_graph(image, nodes, edges)
