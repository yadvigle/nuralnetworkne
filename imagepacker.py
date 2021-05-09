import os
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np


def preprocessor_image(path, new_size=False, to_grey=False):
    image = cv2.imread(path)
    if to_grey:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if new_size:
        image = cv2.resize(image, new_size)
    image = image / 255.
    plt.imshow(image, cmap='viridis')
    return image


def get_meta(path):
    with open(path, 'rb') as f:
        meta = f.readlines()[0]
        meta = np.array([meta])
    return meta


def data_loader(path, file_name, count=5):
    ready_names = list()
    x = list()
    y = list()
    for i, item in enumerate(os.listdir(path)):
        name = item.split('.')[0]
        if name not in ready_names:
            meta_name = name + '.txt'
            image_name = name + '.jpg'
            full_image_path = os.path.join(path, image_name)
            full_meta_path = os.path.join(path, meta_name)
            image_array = preprocessor_image(full_image_path)
            image_meta = get_meta(full_meta_path)
            image_array = np.array(image_array, dtype='uint8')
            x.append(image_array)
            y.append(image_meta)
            ready_names.append(name)
            if i == count:
                break
    x = np.array(x)
    y = np.array(y)
    with open(file_name + '_X.pickle', 'wb') as file:
        pickle.dump(x, file)
    with open(file_name + '_Y.pickle', 'wb') as file:
        pickle.dump(y, file)
    return 0


if __name__ == '__main__':
    path = '/home/vetrow/nnsecondbreath/cats_dogs_dataset/valid'
    print(data_loader(path, 'test'))
