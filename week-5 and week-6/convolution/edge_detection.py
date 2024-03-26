import math
import sys

from PIL import Image, ImageFilter

# Ensure correct usage
if len(sys.argv) != 2:
    sys.exit("Usage: python filter.py filename")

# Open image
image = Image.open(sys.argv[1]).convert("RGB")

# Define the Laplacian kernel
laplacian_kernel = [0, 1, 0,
                    1, -4, 1,
                    0, 1, 0]

# Define the Laplacian of Gaussian (LoG) kernel
log_kernel = [-1, -1, -1,
              -1, 8, -1,
              -1, -1, -1]

# Filter image according to edge detection kernel
filtered = image.filter(ImageFilter.Kernel(
    size=(3, 3),
    kernel=laplacian_kernel,
    scale=1
))

# Show resulting image
filtered.show()
