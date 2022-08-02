from PIL import Image
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

'''
Converts all images in a directory to '.npy' format.
Use np.save and np.load to save and load the images.
Use it for training your neural networks in ML/DL projects. 
'''

# Path to image directory
path = "new_imgs/"
dirs = os.listdir(path)
dirs.sort()

def get_lts_values():
    with open('lts_values.txt', 'r+') as f:
        arr = f.readlines()
        values = [int(line.split(', ')[1].strip()) for line in arr]
        f.close()
    return values

d = get_lts_values()

"""
    The file {x}.npy only contains the RGB value for valid IDs between [x, x + 99] inclusive
    Each row is formatted as [ID, 320 x 320 x 3 ndarray, LTS value]
"""
def create_npy(start, end):
    arr = []
    # Append images to a list
    print(start)
    for i in range(start, end):
        if(os.path.isfile(path + f'{i}.jpg')):
            image = Image.open(path + f'{i}.jpg').convert("RGB")
            image_pillow = np.array(image)
            arr.append(np.array([i, image_pillow, d[i]])) 
        else:
            arr.append(np.array([i, np.zeros((320, 320, 3)), -1]))
    return np.array(arr)

# Sanity check, should return ---89, 320, 1-4
def load_npy(num):
    arr = np.load(f'npy_files/{num}.npy', allow_pickle=True)
    print(len(arr))
    print(arr[86][0])
    print(len(arr[86][1]))
    print(arr[86][2])

if __name__ == "__main__":
    for i in range(0, 40000, 100):
        start = i
        end = min([i + 100, 40000])
        # Convert and save the list of images in '.npy' format
        np.save(f"npy_files/{i}.npy", create_npy(start, end))
    #load_npy(40600)