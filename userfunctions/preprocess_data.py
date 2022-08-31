import argparse
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.data_preprocessing import read_and_return_image_and_mask_gdal, cropped_set_interseks_img_mask



if __name__ == '__main__':
    print("Reading and Patching data")
    parser = argparse.ArgumentParser('-p', '--path', help='Path in which input images and annotations folder lie', default = './AML_project/')
    parser = argparse.ArgumentParser('-p_out', '--path_output', help='Path in which to store the patched data', default = './AML_project/')
    parser = argparse.ArgumentParser('-hp', '--hpatchsize', help='Patch height', default = 64, required=True)
    parser = argparse.ArgumentParser('-wp', '--wpatchsize', help='Patch width', default = 64, required=True)
    parser = argparse.ArgumentParser('-pad', '--padding', help='Whether padding should be used', default = True)
    parser = argparse.ArgumentParser('-h_inter', '--horizontal_intersection', help='Amount of intersecting pixels in horizontal direction', default = 0)
    parser = argparse.ArgumentParser('-v_inter', '--vertical_intersection', help='Amount of intersecting pixels in vertical direction', default = 0)
    
    parser.add_argument()

    args = parser.parse_args()

    cropped_set_interseks_img_mask(read_and_return_image_and_mask_gdal(args.path),
        args.hpatchsize, args.wpatchsize,
        args.padding, args.horizontal_intersection, args.vertical_intersection, 
        args.path_outout)
