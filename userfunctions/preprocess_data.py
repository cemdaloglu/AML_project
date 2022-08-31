import argparse
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.data_preprocessing import read_and_return_image_and_mask_gdal, cropped_set_interseks_img_mask

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    print("Reading and Patching data")
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path in which input images and annotations folder lie', default = '../AML_project')
    parser.add_argument('-p_out', '--path_output', help='Path in which to store the patched data', default = '../AML_project')
    parser.add_argument('-hp', '--hpatchsize', help='Patch height', default = 64, required=True, type=int)
    parser.add_argument('-wp', '--wpatchsize', help='Patch width', default = 64, required=True, type=int)
    parser.add_argument('-pad', '--padding', help='Whether padding should be used', default = False, type=str2bool)
    parser.add_argument('-h_inter', '--horizontal_intersection', help='Amount of intersecting pixels in horizontal direction', default = 0, type=int)
    parser.add_argument('-v_inter', '--vertical_intersection', help='Amount of intersecting pixels in vertical direction', default = 0, type=int)
    parser.add_argument('-t', '--thresh', help='Threshold for cut off of  satellite images', default = 3558, type=int)
    parser.add_argument('-i', '--use_infra', help='Whether 4th infrared channel should be used', default = True, type=str2bool)

    args = parser.parse_args()
    print(f'''
    path = {args.path},
    path_output = {args.path_output}, 
    hpatchsize = {args.hpatchsize}
    wpatchsize =  {args.wpatchsize},
    padding =  {args.padding}, 
    horizontal_intersection = {args.horizontal_intersection}, 
    vertical_intersection = {args.vertical_intersection}, 
    thresh = {args.thresh}, 
    use_infra = {args.use_infra}
    ''' )

    cropped_set_interseks_img_mask(args.path,
        args.hpatchsize, args.wpatchsize,
        args.padding, args.horizontal_intersection, args.vertical_intersection, 
        args.path_output, args.thresh, args.use_infra)
