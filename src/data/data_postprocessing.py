import os
import re
from glob import glob
import numpy as np
import sys



def put_patches_together(results_folder:str, model_name:str, save_path:str = None):
    if save_path is not None:
        model_save_path = os.path.join(save_path, model_name) 
        print("saving prediction to", model_save_path)
        
    pred_path = os.path.join(results_folder, model_name, "evaluation_images/")
    if not os.path.exists(pred_path):
        sys.exit("Run the test.py module first!")
    
    if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

    for city_ind in range(2):
        max_ind_row = 0; max_ind_col = 0   

        searchList = sorted(glob(pred_path+"pred_"+str(city_ind)+"_*"))
    
        max_ind_row = 0     # Find amount of rows
        for item in searchList:
            img_ind = int(re.findall(r'(?!pred?\_'+str(city_ind)+'\_)(\d{1,2})_', item)[-1])
            if img_ind > max_ind_row: 
                max_ind_row = img_ind
        

        max_ind_col = 0     # Find amount of columns
        for item in searchList:
            img_ind = int(re.findall(r'(\d{1,2})', item)[-1])
            if img_ind > max_ind_col: 
                max_ind_col = img_ind


        all_patches = []

        for row in range(max_ind_row):
            col_patches = []
            for col in range(max_ind_col):
                # read in file if exists
                patch_file = os.path.join(pred_path, 'pred_'+str(city_ind)+'_' + str(row) + '_' + str(col) + '.npy')
                if os.path.exists(patch_file):
                    patch = np.load(patch_file)
                    col_patches.append(patch)
                else: # add zero patch for unknown patches which were discarded before
                    patch = np.zeros((128,128))
                    col_patches.append(patch)
            row_stack = np.vstack(col_patches)
            all_patches.append(row_stack)
        restored_img = np.hstack(all_patches)

        if save_path is not None:
            # save restored image
            np.save(model_save_path+"/pred_restored_"+str(city_ind)+".npy", restored_img)


def create_difference(img_groundtruth_pred_path:str, model_name:str):
    '''
    Creates the differences of the groundtruth prediction and the merged prdicion of a certain model. 
    @param img_groundtruth_pred_path: path to where the image, groundtruth and prediction lie of the merged patches
    @param model_name: which model did the prediction 
    '''

    model_save_path = os.path.join(img_groundtruth_pred_path, model_name) 
    print("model_save_path = ", model_save_path)

    for city_ind in range(2):
        groundtruth_path = os.path.join(img_groundtruth_pred_path, 'groundtruth_'+str(city_ind)+'.npy')
        prediction_path = os.path.join(img_groundtruth_pred_path, model_name, 'pred_restored_'+str(city_ind)+'.npy')
        print("groundtruth_path = ", groundtruth_path)
        print("prediction_path = ", prediction_path)

        # read in data 
        groundtruth = np.load(groundtruth_path)
        prediction = np.load(prediction_path)

        # Create differece to groundtruth
        rows_missing = groundtruth.shape[0] - prediction.shape[0]
        cols_missing = groundtruth.shape[1] - prediction.shape[1]

        # bring to same size
        padded_prediction = np.pad(prediction, ((0, rows_missing), (0, cols_missing)), 'constant')

        # set padded prediction to zero where hroundtruth was zero (unlabeled)
        padded_prediction = np.where(groundtruth==0, groundtruth, padded_prediction)

        diff = groundtruth - padded_prediction != 0

        # save restored image
        np.save(model_save_path+"/difference_"+str(city_ind)+".npy", diff)

