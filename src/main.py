from actions import apply_action, revert_action, sample_action
from images import convert_pdf_to_images, vertically_append_images
from detect_layout import detect_layout_elements
from models import AddNetwork, CriticNetwork, PolicyNetwork, RemoveNetwork
from ocr import ocr_elements
from graphs import create_graph, EMBEDDING_DIM, update_coordinates_and_merge_graphs
from visuals import visualize_graph


pdf_path = "docs/examples/1706.03762v7.pdf"
images = convert_pdf_to_images(pdf_path)

images = images[:2]  # Process only the first page for now

# Example usage
layout_boxes = [detect_layout_elements(image) for image in images]
ocr_results = [ocr_elements(image, boxes) for image, boxes in zip(images, layout_boxes)]
graphs_nodes_edges = [create_graph(elements) for elements in ocr_results]

# Merge images and update coordinates
merged_image = vertically_append_images(images)
merged_graph, updated_nodes, updated_edges = update_coordinates_and_merge_graphs(graphs_nodes_edges, images)

input_dim = EMBEDDING_DIM
hidden_dim = 64
projection_dim = 32

# Initialize networks
add_net = AddNetwork(input_dim, hidden_dim)
remove_net = RemoveNetwork(input_dim, hidden_dim)
policy_net = PolicyNetwork(input_dim, hidden_dim)
critic_net = CriticNetwork(input_dim, hidden_dim, projection_dim)

# Forward pass for policy network (example with graph state only)
action_probs = policy_net(merged_graph.x, merged_graph.edge_index)
action_name, action, prob, value = sample_action(merged_graph, add_net, remove_net, critic_net, action_probs)
print(f"Action: {action_name}, Pair: {action}, Probability: {prob}, Value: {value}")


match action_name:
    case "add":
        merged_graph, updated_nodes, updated_edges = apply_action(
            merged_graph, updated_nodes, updated_edges, action_name, action, prob
        )
        merged_graph, updated_nodes, updated_edges = revert_action(
            merged_graph, updated_nodes, updated_edges, action_name, action, prob
        )
    case "remove":
        merged_graph, updated_nodes, updated_edges = apply_action(
            merged_graph, updated_nodes, updated_edges, action_name, action, prob
        )
        reverted_graph, reverted_nodes, reverted_edges = revert_action(
            merged_graph, updated_nodes, updated_edges, action_name, action, prob
        )
    case "stop":
        print("Stopping the layout analysis")
    case _:
        print("Invalid action")
        raise NotImplementedError("Invalid action")


# Visualize the graph
visualize_graph(merged_image, updated_nodes, updated_edges)
