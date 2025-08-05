import os
import cv2
import numpy as np

def image_txt_pairs(folder):
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            base = os.path.splitext(filename)[0]
            jpg_path = os.path.join(folder, f"{base}.jpg")
            txt_path = os.path.join(folder, f"{base}.txt")
            if os.path.isfile(txt_path):
                yield jpg_path, txt_path

# txt file contains list of bounding boxes like this:
# """6 0.8207419898819561 0.6780420860018298 0.1238617200674536 0.1502287282708142
# 0 0.04064080944350759 0.018023787740164682 0.66429173693086 0.1695333943275389
# 1 0.03309443507588533 0.1969807868252516 0.6981450252951096 0.09295516925892038
# 2 0.05383642495784149 0.4853613906678865 0.5043844856661045 0.07346752058554433
# 3 0.7063237774030354 0.34775846294602014 0.2618887015177065 0.1404391582799634
# 4 0.2472175379426644 0.8473924977127173 0.1431281618887015 0.13430924062214086
# 5 0.7235244519392917 0.05818847209515096 0.2747048903878584 0.11866422689844466"""
# Here 0,1,2,3,4,5,6 are the class id, rest columns are xmin, ymin, width, height.
# I want to draw these bounding boxes over the jpg image. Each class will have different color, pre decided. Since there are 7 classes , so 7 differently colored bounding box will be there.

# Define colors for 7 classes (BGR format)
colors = [
    (255, 0, 0),    # Class 0 - Blue
    (0, 255, 0),    # Class 1 - Green
    (0, 0, 255),    # Class 2 - Red
    (255, 255, 0),  # Class 3 - Cyan
    (255, 0, 255),  # Class 4 - Magenta
    (0, 255, 255),  # Class 5 - Yellow
    (128, 0, 128),  # Class 6 - Purple
]

for jpg, txt in image_txt_pairs('./Images'):
    print(jpg, txt)
    img = cv2.imread(jpg)
    h, w = img.shape[:2]

    with open(txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, bw, bh = map(float, parts)
            cls = int(cls)
            # Convert normalized coordinates to pixel values
            cx, cy, bw, bh = x * w, y * h, bw * w, bh * h
            x1 = int(cx)
            y1 = int(cy)
            x2 = int(cx + bw)
            y2 = int(cy + bh)
            color = colors[cls % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, str(cls), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Save or display the image
    out_path = jpg.replace('.jpg', '_bbox.jpg')
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")