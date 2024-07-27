from convert2image import convert_pdf_to_images
from detect_layout import detect_layout_elements
from ocr import ocr_elements
from create_graph import create_graph
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def visualize_graph(image_pil, nodes, edges):
    draw = ImageDraw.Draw(image_pil)

    # Draw bounding boxes
    for node in nodes:
        box = node['bbox']
        draw.rectangle(box, outline="red", width=2)

    # Draw edges
    for edge in edges:
        box_i = nodes[edge[0]]['bbox']
        box_j = nodes[edge[1]]['bbox']
        center_i = ((box_i[0] + box_i[2]) / 2, (box_i[1] + box_i[3]) / 2)
        center_j = ((box_j[0] + box_j[2]) / 2, (box_j[1] + box_j[3]) / 2)
        draw.line([center_i, center_j], fill="green", width=5)

    plt.figure(figsize=(12, 12))
    plt.imshow(image_pil)
    plt.axis('off')
    plt.show()


pdf_path = 'docs/examples/1706.03762v7.pdf'
images = convert_pdf_to_images(pdf_path)

images = images[:1]  # Process only the first page for now

layout_boxes = [detect_layout_elements(image) for image in images]
ocr_results = [ocr_elements(image, boxes) for image, boxes in zip(images, layout_boxes)]
graphs_and_nodes_edges = [create_graph(elements) for elements in ocr_results]


# Now you have graphs for each page in the PDF with text embeddings on the nodes
# and you can visualize one graph
for (graph, nodes, edges), image in zip(graphs_and_nodes_edges, images):
    visualize_graph(image, nodes, edges)
    break  # Visualize only the first page for now