import argparse
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.helpers.visualize import plot_groundtruth_bestpred_differences, plot_worst_segmentations

if __name__ == '__main__':
    print("Reading and Patching data")
    parser = argparse.ArgumentParser()
    parser.add_argument('-best', '--best_model', help='The model which had the highest accuracy to show the entire city prediction', default = "unet_CEL_lr001_hflip_vflip03_wd01", type=str)
    parser.add_argument('-m_list', '--model_list', help='List of length 4 with the U-Net model, the VGG16 model, VGG16 pretrained model and VGG16 index model', nargs="*" , type=str, action='store', dest='model_list', default = ["unet_CEL_lr001_hflip_vflip03_wd01", "unet_CEL_lr001_hflip_vflip03_wd01", "unet_CEL_lr001_hflip_vflip03_wd01", "unet_CEL_lr001_hflip_vflip03_wd01"])
    parser.add_argument('-patch_dir', '--patch_test_mask_dir', help='Directory where the groundtruth patched masks lie of the test set.', default = "patches/test/masks")
    parser.add_argument('-p', '--result_path', help='Path in which results are stored', required=False, default="src/results")
    parser.add_argument('-p_save', '--save_path', help='Path in which to store the merged prediction. "None" if it should not be saved.', default = '../AML_project/img_groundtruth_pred')

    args = parser.parse_args()
    
    print(f'''
    best_model = {args.best_model},
    save_path = {args.save_path}, 
    model_list = {args.model_list}
    ''' )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    plot_groundtruth_bestpred_differences("Frankfurt", args.best_model, args.model_list, args.save_path)
    plot_groundtruth_bestpred_differences("Heidelberg", args.best_model, args.model_list, args.save_path)
    plot_worst_segmentations(args.patch_test_mask_dir, args.result_path, args.best_model)
