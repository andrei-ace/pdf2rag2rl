import cv2
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

OFFSET = 10
CIRCLE_RADIUS = 5


def visualize_graph(image_pil, nodes, edges):
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Draw bounding boxes and node indices
    for idx, node in enumerate(nodes):
        box = node["bbox"]
        # Draw rectangle
        cv2.rectangle(
            image_cv, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2
        )
        # Calculate the position for the text (upper right corner)
        text_position = (box[2], box[1] + 20)  # Adjust the offset as needed
        # Put text at the calculated position
        cv2.putText(
            image_cv,
            str(idx),
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Draw edges without duplication
    drawn_edges = set()
    for edge in edges:
        if (edge[1], edge[0]) in drawn_edges:
            continue  # Skip if the reverse edge is already drawn

        box_i = nodes[edge[0]]["bbox"]
        box_j = nodes[edge[1]]["bbox"]
        center_i = ((box_i[0] + box_i[2]) // 2, (box_i[1] + box_i[3]) // 2)
        center_j = ((box_j[0] + box_j[2]) // 2, (box_j[1] + box_j[3]) // 2)

        # Add small random offset to avoid overlapping lines
        offset_i = (random.randint(-OFFSET, OFFSET), random.randint(-OFFSET, OFFSET))
        offset_j = (random.randint(-OFFSET, OFFSET), random.randint(-OFFSET, OFFSET))
        center_i = (center_i[0] + offset_i[0], center_i[1] + offset_i[1])
        center_j = (center_j[0] + offset_j[0], center_j[1] + offset_j[1])

        # Draw the line
        cv2.line(image_cv, center_i, center_j, color=(0, 255, 0), thickness=2)

        # Draw dots at the ends of the edges
        cv2.circle(
            image_cv, center_i, radius=CIRCLE_RADIUS, color=(255, 0, 0), thickness=-1
        )
        cv2.circle(
            image_cv, center_j, radius=CIRCLE_RADIUS, color=(255, 0, 0), thickness=-1
        )

        # Mark this edge as drawn
        drawn_edges.add((edge[0], edge[1]))

    # Convert OpenCV image back to PIL format
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    # Display the image using matplotlib
    plt.figure(figsize=(12, 12))
    plt.imshow(image_pil)
    plt.axis("off")
    plt.show()
