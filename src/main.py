import torch

from images import convert_pdf_to_images, vertically_append_images
from detect_layout import detect_layout_elements
from ocr import ocr_elements
from graphs import create_graph, EMBEDDING_DIM, update_coordinates_and_merge_graphs
from ppo import PPO
from visuals import visualize_graph


pdf_path = "docs/examples/1706.03762v7.pdf"
images = convert_pdf_to_images(pdf_path)

images = images[:1]  # Process only the first page for now

# Example usage
layout_boxes = [detect_layout_elements(image) for image in images]
ocr_results = [ocr_elements(image, boxes) for image, boxes in zip(images, layout_boxes)]
graphs_nodes_edges = [create_graph(elements) for elements in ocr_results]

# Merge images and update coordinates
merged_image = vertically_append_images(images)
merged_graph, merged_nodes, merged_edges = update_coordinates_and_merge_graphs(graphs_nodes_edges, images)

ppo = PPO("cuda" if torch.cuda.is_available() else "cpu")
ppo.episode(merged_graph, merged_nodes, merged_edges)
trajectory, merged_graph, merged_nodes, merged_edges = ppo.infer_trajectory(merged_graph, merged_nodes, merged_edges)
print(trajectory)

visualize_graph(merged_image, merged_nodes, merged_edges)
