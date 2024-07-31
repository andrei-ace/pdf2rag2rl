import os
import shutil
import torch
import joblib

from images import convert_pdf_to_images, vertically_append_images
from detect_layout import detect_layout_elements
from ocr import ocr_elements
from graphs import create_graph, extract_text_from_graph, split_graph, update_coordinates_and_merge_graphs
from ppo import PPO
from questions import PDFS
from rag import rag
from visuals import visualize_graph

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

EPOCHS = 4
# Define a cache directory
CACHE_DIR = '__cache__'

# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_results(cache_key, func, *args, **kwargs):
    # Generate a cache file path
    cache_file = os.path.join(CACHE_DIR, f'{cache_key}.pkl')
    
    if os.path.exists(cache_file):
        # Load results from cache
        return joblib.load(cache_file)
    else:
        # Compute the results and cache them
        results = func(*args, **kwargs)
        joblib.dump(results, cache_file)
        return results
    
# Function to process the PDF and return the results
def process_pdf(pdf_path):
    images = convert_pdf_to_images(pdf_path)
    layout_boxes = [detect_layout_elements(image) for image in images]
    ocr_results = [ocr_elements(image, boxes) for image, boxes in zip(images, layout_boxes)]
    graphs_nodes_edges = [create_graph(elements) for elements in ocr_results]
    merged_graph, merged_nodes, merged_edges = update_coordinates_and_merge_graphs(graphs_nodes_edges, images)
    merged_image = vertically_append_images(images)
    return merged_graph, merged_nodes, merged_edges, merged_image

def infer_pdf(pdf_entry, ppo, device=device):
    (pdf_path, questions_answers) = pdf_entry
    cache_key = os.path.basename(pdf_path)  # or generate a unique key based on pdf_path
    merged_graph, merged_nodes, merged_edges, merged_image = cache_results(cache_key, process_pdf, pdf_path)
    merged_graph = merged_graph.to(device)
    
    save_path = "docs/output/no_trainig.png"
    if ppo is not None:
        # This will change the graph in place r
        trajectory, merged_graph, merged_nodes, merged_edges = ppo.infer_trajectory(
            merged_graph, merged_nodes, merged_edges
        )
        save_path = "docs/output/with_trainig.png"
    visualize_graph(merged_image, merged_nodes, merged_edges, save_path=save_path)
    results = rag(merged_graph, merged_nodes, merged_edges, questions_answers)
    for question, answer, score in results:
        print(f"Question: {question}\nGenerated Answer: {answer}\nScore: {score}\n")
    mean_score = sum([score for _, _, score in results]) / len(results)
    return mean_score


def train_pdf(pdf_entry, ppo, device=device):
    (pdf_path, questions_answers) = pdf_entry
    cache_key = os.path.basename(pdf_path)
    merged_graph, merged_nodes, merged_edges, _ = cache_results(cache_key, process_pdf, pdf_path)
    merged_graph = merged_graph.to(device)
    ppo.episode(merged_graph, merged_nodes, merged_edges, questions_answers)


print(f"Using device info: {torch.cuda.get_device_properties(0)}")

ppo = PPO(device=device)
for _ in range(EPOCHS):
    for pdf_entry in PDFS:
        train_pdf(pdf_entry, ppo)

if os.path.exists("docs/output"):
    shutil.rmtree("docs/output")
os.makedirs("docs/output", exist_ok=True)
mean_score_notrain = infer_pdf(PDFS[0], None)
mean_score_withtrain = infer_pdf(PDFS[0], ppo)

print(f"Mean scores no training: {mean_score_notrain:.4f} vs with trainig: {mean_score_withtrain:.4f}\n")


# subgraphs = split_graph(merged_graph, merged_nodes, merged_edges)

# # delete and recreate the output directory
# if os.path.exists("docs/output"):
#     shutil.rmtree("docs/output")
# os.makedirs("docs/output", exist_ok=True)

# for i, (subgraph, nodes, edges) in enumerate(subgraphs):
#     visualize_graph(merged_image, nodes, edges, save_path=f"docs/output/subgraph-{i}.png")

# visualize_graph(merged_image, merged_nodes, merged_edges, save_path="docs/output/all.png")
