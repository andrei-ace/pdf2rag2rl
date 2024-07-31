import re
import pytesseract


def ocr_elements(image_pil, boxes):
    elements = []
    for box in boxes:
        x0, y0, x1, y1 = map(int, box)
        cropped_img = image_pil.crop((x0, y0, x1, y1))
        text = pytesseract.image_to_string(cropped_img)
        text = re.sub(r'[^\x20-\x7E]', '', text)  # Keep only printable ASCII characters        
        elements.append(((x0, y0, x1, y1), text))
    return elements
