from tqdm import tqdm
from PIL import Image
import numpy as np
import os

def read_idx3_ubyte(file_path):

    with open(file_path, 'rb') as file:
        magic_number = int.from_bytes(file.read(4), 'big')
        num_images = int.from_bytes(file.read(4), 'big')
        num_rows = int.from_bytes(file.read(4), 'big')
        num_cols = int.from_bytes(file.read(4), 'big')


        images = np.empty((num_images, num_rows, num_cols), dtype=np.uint8)
        
        for i in tqdm(range(num_images), desc="Reading images"):
            images[i] = np.frombuffer(file.read(num_rows * num_cols), dtype=np.uint8).reshape(num_rows, num_cols)
    
    return images

def save_images(images, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i, image in enumerate(images):
        img = Image.fromarray(image, mode='L')
        img.save(os.path.join(output_dir, f'image_{i}.png'))

output_dir = "Z:/OpenSourceContribution/neuralDigits/NeuroDigits/Dataset"

images0 = read_idx3_ubyte("Z:/OpenSourceContribution/neuralDigits/NeuroDigits/archive (1)/train-images.idx3-ubyte")
save_images(images0, output_dir)

print(f"Saved {len(images0)} images to {output_dir}")

images1 = read_idx3_ubyte("Z:/OpenSourceContribution/neuralDigits/NeuroDigits/archive (1)/t10k-images.idx3-ubyte")
save_images(images1, output_dir)

print(f"Saved {len(images1)} images to {output_dir}")
