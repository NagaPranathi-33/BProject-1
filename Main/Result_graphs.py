# Import necessary libraries
from PIL import Image
import IPython.display as display

# ✅ Correct file paths for Google Colab (assuming images are in /content/graphs/)
image_files = [f"/content/graphs/{i}.jpeg" for i in range(1, 13)]

# ✅ Display all images
for img_path in image_files:
    try:
        img = Image.open(img_path)
        display.display(img)
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
