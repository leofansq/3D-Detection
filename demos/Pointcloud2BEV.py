import cv2
import numpy as np

from google.protobuf import text_format
from avod.builders.dataset_builder import DatasetBuilder


def points2img(points, bgcolor='black'):
    # Load Image
    image = (points * 255.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Bgcolor: 'black' or 'white'
    if bgcolor == 'white':
        # Image enhancement
        #image = cv2.add(image,image)

        image = cv2.bitwise_not(image)

    return image


def create_bev(dataset, img_num, save_path, bgcolor, save_height_maps=False):

    bev_generator = 'slices'
    slices_config = \
        """
        slices {
            height_lo: -0.2
            height_hi: 2.3
            num_slices: 5
        }
        """ 

    if dataset == "TRAIN":
        dataset_config = DatasetBuilder.copy_config(DatasetBuilder.KITTI_VAL)
    elif dataset == "TEST":
        dataset_config = DatasetBuilder.copy_config(DatasetBuilder.KITTI_TEST)
    else:
        raise ValueError(('Invalid dataset'))

    dataset_config = DatasetBuilder.merge_defaults(dataset_config)

    # Overwrite bev_generator
    if bev_generator == 'slices':
        text_format.Merge(slices_config, dataset_config.kitti_utils_config.bev_generator)
    else:
        raise ValueError('Invalid bev_generator')

    dataset = DatasetBuilder.build_kitti_dataset(dataset_config, use_defaults=False)

    for img_idx in range(img_num):

        sample_name = "{:06}".format(img_idx)
        print('Creating BEV maps for: {} ==='.format(sample_name))

        # Load image
        image = cv2.imread(dataset.get_rgb_image_path(sample_name))
        image_shape = image.shape[0:2]

        kitti_utils = dataset.kitti_utils
        point_cloud = kitti_utils.get_point_cloud('lidar', img_idx, image_shape)
        ground_plane = kitti_utils.get_ground_plane(sample_name)
        bev_images = kitti_utils.create_bev_maps(point_cloud, ground_plane)

        height_maps = np.array(bev_images.get("height_maps"))
        density_map = np.array(bev_images.get("density_map"))
        

        # Height maps if save_height_maps = True
        if save_height_maps:

            for map_idx in range(len(height_maps)):
                height_map = height_maps[map_idx]
                height_map = points2img(height_map, bgcolor)

                cv2.imwrite(save_path + sample_name + "h" + str(map_idx) + ".png", height_map)

        # Density map (Normal BEV)
        density_map = points2img(density_map, bgcolor)

        cv2.imwrite(save_path + sample_name + ".png", density_map)


    cv2.waitKey()


if __name__ == "__main__":
    DATASET = "TRAIN"       # "TRAIN"  OR  "TEST"    
    IMG_NUM = 7481

    SAVE_PATH = "./bev/"
    BGCOLOR = "black"
    SAVE_HEIGHT_MAPS = False

    create_bev(DATASET, IMG_NUM, SAVE_PATH, BGCOLOR, SAVE_HEIGHT_MAPS)
