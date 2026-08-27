import os

dataset_root = "Dataset"

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
)

total_images = 0

print("Searching folder:")
print(os.path.abspath(dataset_root))
print()

for root, folders, files in os.walk(dataset_root):

    for filename in files:

        if filename.lower().endswith(image_extensions):

            image_path = os.path.join(root, filename)

            print(image_path)

            total_images += 1

print("\n==============================")
print("Total images found:", total_images)
print("==============================")