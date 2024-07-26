from convert2image import convert_pdf_to_images
from detect_layout import detect_layout_elements
from ocr import ocr_elements
from create_graph import create_graph

pdf_path = 'docs/examples/1706.03762v7.pdf'
images = convert_pdf_to_images(pdf_path)

layout_boxes = [detect_layout_elements(image) for image in images]
ocr_results = [ocr_elements(image, boxes) for image, boxes in zip(images, layout_boxes)]
graphs = [create_graph(elements) for elements in ocr_results]

# Now you have graphs for each page in the PDF
for graph in graphs:
    print(graph)