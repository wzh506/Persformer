import cv2
import numpy as np
    # Define colors for each value from 0 to 9
colors = {
    0: (0, 0, 0),       # Black
    1: (255, 0, 0),     # Blue
    2: (0, 255, 0),     # Green
    3: (0, 0, 255),     # Red
    4: (255, 255, 0),   # Cyan
    5: (255, 0, 255),   # Magenta
    6: (0, 255, 255),   # Yellow
    7: (128, 0, 0),     # Maroon
    8: (0, 128, 0),     # Dark Green
    9: (0, 0, 128)      # Navy
}
bev_h, bev_w = gt_mask.shape
color_gt_mask = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
for value, color in colors.items():
    color_gt_mask[gt_mask == value] = color

seg_bev = seg_bev_map.squeeze(0)
bev_h, bev_w = seg_bev.shape
color_gt_mask = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
for value, color in colors.items():
    color_gt_mask[seg_bev == value] = color
cv2.imwrite(f"seg_bev_map.jpg", color_gt_mask)

# Save gt_mask as .jpg file
cv2.imwrite("gt_mask.jpg", color_gt_mask)

color_dt_mask = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
for value, color in colors.items():
    color_dt_mask[dt_mask == value] = color

# Save gt_mask as .jpg file
cv2.imwrite("dt_mask.jpg", color_dt_mask)


color_gt_mask = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
for value, color in colors.items():
    color_gt_mask[gt_mask == value] = color

# Save gt_mask as .jpg file
cv2.imwrite("gt_mask.jpg", color_gt_mask)

color_valid_mask = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
for value, color in colors.items():
    color_valid_mask[valid_mask == value] = color

# Save gt_mask as .jpg file
cv2.imwrite("valid_mask.jpg", color_valid_mask)


from PIL import Image
import numpy as np

# Assuming seg_label is a numpy array with shape (360,480) and values in a suitable range (e.g., 0/1)
img = Image.fromarray(seg_label.astype(np.uint8))
img.save("seg_label.png")
print("seg_label saved as seg_label.png")