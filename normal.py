import os
from multiprocessing import Pool, cpu_count

# surface_normal library from https://github.com/valgur/surface-normal
from surface_normal import normals_from_depth
from tqdm.auto import tqdm
from env import KITTI_DATASET_PATH

INTRINSICS = {
    "2011_09_26": (721.5377, 609.5593, 172.8540),
    "2011_09_28": (707.0493, 604.0814, 180.5066),
    "2011_09_29": (718.3351, 600.3891, 181.5122),
    "2011_09_30": (707.0912, 601.8873, 183.1104),
    "2011_10_03": (718.8560, 607.1928, 185.2157),
}

KITTI_GT_PATH = os.path.join(KITTI_DATASET_PATH, 'data_depth_annotated')
KITTI_NORMALS_PATH = os.path.join(KITTI_DATASET_PATH, 'data_depth_normals')
num_cores = cpu_count()

def _process(args):
    normals_from_depth(*args, use_cuda=True)
    return args[1]

if __name__ == '__main__':
    if not os.path.exists(KITTI_NORMALS_PATH):
        os.mkdir(KITTI_NORMALS_PATH)

    processing_args = []
    for split in ['train', 'val']:
        date_folder_list = sorted(os.listdir(os.path.join(KITTI_GT_PATH, split))) # list of 2011_XX_XX_drive_XXXX_sync
        
        for date_folder in date_folder_list:
            sub_path_to_date_folder = os.path.join(split, date_folder, 'proj_depth', 'groundtruth')
            gt_path = os.path.join(KITTI_GT_PATH, sub_path_to_date_folder) # path to groundtruth

            date = date_folder[:10]
            intrinsic = INTRINSICS[date] # get intrincsic of that date

            for img_folder in ['image_02', 'image_03']:
                sub_path_to_img_folder = os.path.join(sub_path_to_date_folder, img_folder) # subpath to image_02 or image_03
                gt_path = os.path.join(KITTI_GT_PATH, sub_path_to_img_folder)

                img_fn_list = sorted(os.listdir(gt_path)) # list of png file
                for img_fn in img_fn_list:
                    normal_folder = os.path.join(KITTI_NORMALS_PATH, sub_path_to_img_folder) # stored directory
                    if not os.path.exists(normal_folder):
                        os.makedirs(normal_folder)

                    gt_img_path = os.path.join(KITTI_GT_PATH, sub_path_to_img_folder, img_fn) # whole path to png files
                    normal_img_path = os.path.join(normal_folder, img_fn)
 
                    if not os.path.exists(normal_img_path):
                        processing_args.append((gt_img_path, normal_img_path, intrinsic, 15, 0.1))

    print("Using", num_cores, "cores")
    pool = Pool(num_cores)
    for _ in tqdm(pool.imap_unordered(_process, processing_args), total=len(processing_args)):
        pass
