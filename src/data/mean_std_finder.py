from .data_preprocessing import read_and_return_image_and_mask_gdal
import numpy as np
import glob


def mean_std_finder(city_image_path):
    cities = ['*karlsruhe*', '*munchen*', '*stuttgart*', '*wurzburg*', '*heilbronn*', '*tubingen*', '*freiburg*',
              '*darmstadt*', '*mainz*', '*heidelberg*', '*frankfurt*']
    train_image_stack = [glob.glob(city_image_path + city)[0] for city in cities]

    mean_std_all = {}
    for idx, image_path in enumerate(train_image_stack):
        img, _ = read_and_return_image_and_mask_gdal(image_path, image_path)
        mean_std_all.update({cities[idx] + "_mean": [np.mean(img[:, :, 0]), np.mean(img[:, :, 1]),
                                                     np.mean(img[:, :, 2]), np.mean(img[:, :, 3])]})
        mean_std_all.update({cities[idx] + "_std": [np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2]),
                                                    np.std(img[:, :, 3])]})
    return mean_std_all
