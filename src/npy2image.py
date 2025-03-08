import numpy as np
from PIL import Image
import sys

def load_data():
    coords = np.load(sys.argv[1])
    image = np.load(sys.argv[2])
    h = int(np.sqrt(coords.shape[0]))  # Assuming square image
    w = h
    return coords, image.reshape(h, w)

def save_image(image, output_file):
    img = Image.fromarray((image * 255).astype(np.uint8), mode="L")
    img.save(output_file)

if __name__ == "__main__":
    coords, image = load_data()
    save_image(image, "output.png")
