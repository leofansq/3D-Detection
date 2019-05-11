import cv2
from google.protobuf import text_format
import numpy as np
import numpy.random as random

from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.visualization import vis_utils

from avod.builders.dataset_builder import DatasetBuilder
from avod.core import box_3d_encoder
from avod.core import box_3d_projector


def draw_boxes(image, gt_boxes_norm, pre_boxes_norm):
    """Draws gt-boxes and pre-boxes on the bev image

    Args:
        image: bev image
        gt_boxes_norm: gt_box corners normalized to the size of the image
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        pre_boxes_norm:pre_box corners normalized to the size of the image
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    Returns:
        The image with boxes drawn on it. If boxes_norm is None,
            returns the original image
    """
    # Load Image
    image = (image * 255.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #image = cv2.add(image,image)
    #image = cv2.bitwise_not(image)
    # Draw prediction boxes
    for pre_box_points in pre_boxes_norm:
        image_shape = np.flip(image.shape[0:2], axis=0)

        for pre_box_point_idx in range(len(pre_box_points)):

            pre_start_point = pre_box_points[pre_box_point_idx] * image_shape
            pre_end_point = pre_box_points[(pre_box_point_idx + 1) % 4] * image_shape

            pre_start_point = pre_start_point.astype(np.int32)
            pre_end_point = pre_end_point.astype(np.int32)

            cv2.line(
                image, tuple(pre_start_point),
                tuple(pre_end_point),
                (107,222,35), thickness=1)

    # Draw boxes if they exist
    if gt_boxes_norm is not None:
        for gt_box_points in gt_boxes_norm:
            for gt_box_point_idx in range(len(gt_box_points)):

                gt_start_point = gt_box_points[gt_box_point_idx] * image_shape
                gt_end_point = gt_box_points[(gt_box_point_idx + 1) % 4] * image_shape

                gt_start_point = gt_start_point.astype(np.int32)
                gt_end_point = gt_end_point.astype(np.int32)

                cv2.line(
                    image, tuple(gt_start_point),
                    tuple(gt_end_point),
                    (0,0,205), thickness=1)

    return image


def main():
    """
    Displays the BEV results of a sample.
    gt_boxes are in red.
    pre_boxes are in green.
    """
    ##############################
    # Options
    ##############################

    bev_generator = 'slices'
    slices_config = \
        """
        slices {
            height_lo: -0.2
            height_hi: 2.3
            num_slices: 5
        }
        """
    # Use None for a random image
    #img_idx = None
    img_idx = 6

    show_ground_truth = True  # Whether to overlay ground_truth boxes
    show_height_maps = False  # Whether to show the five height maps
    show_images = False       # Whether to show the images

    point_cloud_source = 'lidar'
    pre_label_dir = '/home/cecilia/leo_projects/bishe2019/3D-Detection/avod/data/outputs/pyramid_cars_with_aug_rep_loss/predictions/kitti_native_eval/0.1/112000/data/'
    ##############################
    # End of Options
    ##############################

    dataset_config = DatasetBuilder.copy_config(DatasetBuilder.KITTI_VAL)
    dataset_config = DatasetBuilder.merge_defaults(dataset_config)

    # Overwrite bev_generator
    if bev_generator == 'slices':
        text_format.Merge(slices_config,
                          dataset_config.kitti_utils_config.bev_generator)
    else:
        raise ValueError('Invalid bev_generator')

    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)

    if img_idx is None:
        img_idx = int(random.random() * dataset.num_samples)

    sample_name = "{:06}".format(img_idx)
    print('=== Showing BEV maps for image: {}.png ==='.format(sample_name))

    # Load image
    image = cv2.imread(dataset.get_rgb_image_path(sample_name))
    image_shape = image.shape[0:2]

    kitti_utils = dataset.kitti_utils
    point_cloud = kitti_utils.get_point_cloud(
        point_cloud_source, img_idx, image_shape)
    ground_plane = kitti_utils.get_ground_plane(sample_name)
    bev_images = kitti_utils.create_bev_maps(point_cloud, ground_plane)

    height_maps = np.array(bev_images.get("height_maps"))
    density_map = np.array(bev_images.get("density_map"))

    # Get groundtruth bev-info
    gt_box_points, gt_box_points_norm = [None, None]
    if show_ground_truth:
        gt_obj_labels = obj_utils.read_labels(dataset.label_dir, img_idx)
        gt_filtered_objs = gt_obj_labels

        gt_label_boxes = []
        for gt_label in gt_filtered_objs:
            gt_box = box_3d_encoder.object_label_to_box_3d(gt_label)
            gt_label_boxes.append(gt_box)

        gt_label_boxes = np.array(gt_label_boxes)
        gt_box_points, gt_box_points_norm = box_3d_projector.project_to_bev(
            gt_label_boxes, [[-40, 40], [0, 70]])

    # Get prediction bev-info
    pre_box_points, pre_box_points_norm = [None, None]
    pre_obj_labels = obj_utils.read_labels(pre_label_dir, img_idx)

    pre_filtered_objs = pre_obj_labels

    pre_label_boxes = []
    for pre_label in pre_filtered_objs:
        pre_box = box_3d_encoder.object_label_to_box_3d(pre_label)
        pre_label_boxes.append(pre_box)

    pre_label_boxes = np.array(pre_label_boxes)
    pre_box_points, pre_box_points_norm = box_3d_projector.project_to_bev(
        pre_label_boxes, [[-40, 40], [0, 70]])

    
    rgb_img_size = (np.array((1242, 375)) * 0.75).astype(np.int16)
    img_x_start = 60
    img_y_start = 330

    img_x = img_x_start
    img_y = img_y_start
    img_w = 400
    img_h = 350
    img_titlebar_h = 20

    # Show images if show_images = True
    if show_images:
        vis_utils.cv2_show_image("Image", image,
                                 size_wh=rgb_img_size, location_xy=(img_x, 0))

    # Height maps if show_height_maps = True
    if show_height_maps:

        for map_idx in range(len(height_maps)):
            height_map = height_maps[map_idx]

            height_map = draw_boxes(height_map, gt_box_points_norm, pre_box_points_norm)
            vis_utils.cv2_show_image(
                "Height Map {}".format(map_idx), height_map, size_wh=(
                    img_w, img_h), location_xy=(
                    img_x, img_y))

            img_x += img_w
            # Wrap around
            if (img_x + img_w) > 1920:
                img_x = img_x_start
                img_y += img_h + img_titlebar_h

    # Density map (Normal BEV)
    density_map = draw_boxes(density_map, gt_box_points_norm, pre_box_points_norm)
    vis_utils.cv2_show_image(
        "Density Map", density_map, size_wh=(
            img_w, img_h), location_xy=(
            img_x, img_y))

    cv2.waitKey()


if __name__ == "__main__":
    main()
