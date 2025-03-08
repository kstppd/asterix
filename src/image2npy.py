import numpy as np
import sys
from PIL import Image

def read_image(file):
    img = Image.open(file).convert("L")  # Convert to grayscale to ensure 2D array
    return np.array(img, dtype=np.float32) / 255.0

def extract_coords(image):
    h, w = image.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords =-0.5+ np.stack([x.ravel(), y.ravel()], axis=1,dtype=np.float32)/(h-1)
    return coords



if __name__ == "__main__":
    file = sys.argv[1]
    image = read_image(file)
    coords=extract_coords(image)
    image=image.flatten()
    np.save("x.npy",coords)
    np.save("y.npy",image)
