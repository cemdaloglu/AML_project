import os
import re
from glob import glob
import numpy as np
import sys



def put_patches_together(results_folder:str, model_name:str, save_path:str = None):
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

