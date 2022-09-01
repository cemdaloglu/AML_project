import glob
import numpy as np


def dataset_prepare(train_set_path: str, test_set_path: str, train_label_path: str, test_label_path: str,
                    patch_size: np.array) -> list:
    train_set = []
    for image_path in glob.glob(f"{train_set_path}/*.npy"):
        train_set.append(np.load(image_path))

    label_set = []
    for image_path in glob.glob(f"{train_label_path}/*.npy"):
        label_set.append(np.load(image_path))

    test_set = []
    for image_path in glob.glob(f"{test_set_path}/*.npy"):
        test_set.append(np.load(image_path))

    test_label_set = []
    for image_path in glob.glob(f"{test_label_path}/*.npy"):
        test_label_set.append(np.load(image_path))

    train_set = np.reshape(train_set, (-1, 4, patch_size[0], patch_size[1]))
    label_set = np.reshape(label_set, (-1, patch_size[0], patch_size[1]))
    test_set = np.reshape(test_set, (-1, 4, patch_size[0], patch_size[1]))
    test_label_set = np.reshape(test_label_set, (-1, patch_size[0], patch_size[1]))

    return train_set, label_set, test_set, test_label_set