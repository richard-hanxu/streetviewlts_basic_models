import numpy as np
import random

"""
    IDs are saved as an n x 1 ndarray in each npy file (ID, LTS value)
    Each file contains the valid IDs for each LTS 
"""
def make_lts_files():
    lts_lists = [[], [], [], []]
    with open('lts_values.txt', 'r+') as f:
        arr = f.readlines()
        ids = [int(line.split(',')[0].strip()) for line in arr]
        lts = [int(line.split(',')[1].strip()) for line in arr]
        f.close()
    with open('do_not_use.txt', 'r+') as f:
        do_not_use = [int(line) for line in f.readlines()]
        f.close()
    for i in range(0, len(lts)):
        if ids[i] not in do_not_use:
            lts_lists[lts[i] - 1].append(ids[i])
    for i in range(0, 4):
        np.save(f'lts_{i + 1}.npy', np.array(lts_lists[i]))
"""
    Accesses lts_{1-4}.npy and returns a 4 x n ndarray of IDs
"""
def retrieve_lts_ids(quantities : list[int]):
    ids = []
    for i in range(0, len(quantities)):
        arr = np.load(f'lts_{i + 1}.npy')
        random.shuffle(arr)
        ids.append(np.array(arr)[: quantities[i]])
    #print(len(ids))
    #print(len(ids[0]))
    #print(ids[0][0])
    return np.array(ids)

"""
    Input is a 4 x 3 list
    Returns a 4 x 3 x n array (where n is the number of images in each split) of IDs
"""
def get_ids(splits):
    img_amts = []
    for i in range(0, 4):
        img_amts.append(sum(splits[i]))
    ids = retrieve_lts_ids(img_amts)
    partitioned_ids = []
    for i in range(0, 4):
        training_set = ids[i][: splits[i][0]]
        validation_set = ids[i][splits[i][0] : splits[i][0] + splits[i][1]]
        test_set = ids[i][splits[i][1] : splits[i][1] + splits[i][2]]
        partitioned_ids.append(np.array([training_set, validation_set, test_set]))

    # partitioned_ids is currently a 4 x 3 x n ndarray containing IDs
    # Sanity check, should return 4, 3, number of LTS1 training images, and a single ID respectively
    #print(len(partitioned_ids))
    #print(len(partitioned_ids[0]))
    #print(len(partitioned_ids[0][0]))
    #print(partitioned_ids[0][0][0])
    return partitioned_ids

"""
    Accepts an n x 1 ndarray of IDs, and returns an n x [(1 x 307200), 1] ndarray
"""
def get_batch(ids):
    data = []
    # Iterating through each LTS value
    for id in ids:
        npy_path = f'npy_files/{id - id % 100}.npy'
        rgb_array = np.load(npy_path, allow_pickle=True)[id % 100][1].reshape(307200)
        lts_label = np.load(npy_path, allow_pickle=True)[id % 100][2]
        data.append([rgb_array, lts_label])
    return np.array(data, dtype=object)

def merge_ids(id_dataset, type):
    if type == 'train':
        return np.concatenate((id_dataset[0][0], id_dataset[1][0], id_dataset[2][0], id_dataset[3][0]), axis=0)
    elif type == 'val':
        return np.concatenate((id_dataset[0][1], id_dataset[1][1], id_dataset[2][1], id_dataset[3][1]), axis=0)
    elif type == 'test':
        return np.concatenate((id_dataset[0][2], id_dataset[1][2], id_dataset[2][2], id_dataset[3][2]), axis=0)
    else:
        return 0

if __name__ == '__main__':
    #make_lts_files()
    #print(get_batch([1005, 1006]))
    lts_1_splits = [500, 500, 500]
    lts_2_splits = [500, 500, 500]
    lts_3_splits = [500, 500, 500]
    lts_4_splits = [500, 500, 500]
    splits = [lts_1_splits, lts_2_splits, lts_3_splits, lts_4_splits]
    ids = get_ids(splits)
    print(ids)
    
