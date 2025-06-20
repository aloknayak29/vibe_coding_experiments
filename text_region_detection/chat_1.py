import cv2
import numpy as np

def detect_text_regions(image_path):
    # Load image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply morphological gradient (blackhat or gradient works well for text)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morph_kernel)

    # Binarize using Otsu's method
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Connect horizontally oriented regions
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, connect_kernel)

    # Find contours and filter them
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)

        # Filter based on size and shape
        if area > 100 and 1 < aspect_ratio < 15:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output

# Example usage
output_img = detect_text_regions('path_to_document.jpg')
cv2.imshow("Detected Text Regions", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
