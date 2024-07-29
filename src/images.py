from pdf2image import convert_from_path
from PIL import Image


def convert_pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images


def vertically_append_images(images):
    widths, heights = zip(*(img.size for img in images))
    total_height = sum(heights)
    max_width = max(widths)

    merged_image = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for img in images:
        merged_image.paste(img, (0, y_offset))
        y_offset += img.size[1]

    return merged_image
