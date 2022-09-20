import argparse
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.helpers.bash_helper import str2bool
from src.data.data_postprocessing import put_patches_together, create_difference
from src.helpers.visualize import plot_groundtruth_prediction, plot_groundtruth_bestpred_differences

#TODO: run test.py before!

if __name__ == '__main__':
    print("Reading and Patching data")
    parser = argparse.ArgumentParser()
    parser.add_argument('-res', '--results_folder', help='Path in which predicted patches lie', default = "src/results")
    parser.add_argument('-m_name', '--model_name', help='Name of the trained model', default = "unet_CEL_lr001_hflip_vflip03_wd01", required=True)
    parser.add_argument('-best', '--best_model', help='The model which had the highest accuracy to show the entire city prediction', default = "unet_CEL_lr001_hflip_vflip03_wd01", type=str)
    parser.add_argument('-m_list', '--model_list', help='List of length 4 with the U-Net model, the VGG16 model, VGG16 pretrained model and VGG16 index model', nargs="+" , default = ["unet_CEL_lr001_hflip_vflip03_wd01", "unet_CEL_lr001_hflip_vflip03_wd01", "unet_CEL_lr001_hflip_vflip03_wd01", "unet_CEL_lr001_hflip_vflip03_wd01"], type=str2bool)
    parser.add_argument('-p_save', '--save_path', help='Path in which to store the merged prediction. "None" if it should not be saved.', default = '../AML_project/img_groundtruth_pred')
    parser.add_argument('-s', '--savebool', help='Whether to save the plots, prediction and difference', default = True, type=str2bool)
    

    args = parser.parse_args()
    print(f'''
    results_folder = {args.results_folder},
    model_name = {args.model_name},
    save_path = {args.save_path}
    ''' )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    put_patches_together(args.results_folder, args.model_name, args.save_path)
    create_difference(args.save_path, model_name = args.model_name)
    if args.savebool: 
        #plot_groundtruth_prediction("Frankfurt", args.save_path+"/groundtruth_0.npy", args.save_path+f"/{args.model_name}/pred_restored_0.npy", args.save_path+f"/{args.model_name}")
        #plot_groundtruth_prediction("Heidelberg", args.save_path+"/groundtruth_1.npy", args.save_path+f"/{args.model_name}/pred_restored_1.npy", args.save_path+f"/{args.model_name}")
        plot_groundtruth_bestpred_differences("Frankfurt", args.best_model, args.model_list, args.save_path)
        plot_groundtruth_bestpred_differences("Heidelberg", args.best_model, args.model_list, args.save_path)
