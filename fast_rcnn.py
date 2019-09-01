import tensorflow as tf


def boxes_area(boxes):
    min_x, min_y, max_x, max_y = tf.split(boxes, 4, axis=-1)
    return tf.squeeze((max_x - min_x) * (max_y - min_y), axis=-1)


def pairwise_intersection_area(boxes1, boxes2):
    '''
    :param boxes1: [N, 4]
    :param boxes2: [M, 4]
    :return: [N, M]
    '''
    min_x1, min_y1, max_x1, max_y1 = tf.split(boxes1, 4, axis=-1)  # [N, 1]
    min_x2, min_y2, max_x2, max_y2 = tf.split(boxes2, 4, axis=-1)  # [M, 1]

    minimun_max_y = tf.minimum(max_y1, tf.transpose(max_y2, [1, 0]))
    maximum_min_y = tf.maximum(min_y1, tf.transpose(min_y2, [1, 0]))
    intersecting_h = tf.maximum(0.0, minimun_max_y - maximum_min_y)  # [N, M]

    minimun_max_x = tf.minimum(max_x1, tf.transpose(max_x2, [1, 0]))
    maximum_min_x = tf.maximum(min_x1, tf.transpose(min_x2, [1, 0]))
    intersecting_w = tf.maximum(0.0, minimun_max_x - maximum_min_x)  # [N, M]

    return tf.multiply(intersecting_h, intersecting_w)  # [N, M]


def pairwise_iou(boxes1, boxes2):
    '''
    :param boxes1: [N, 4]
    :param boxes2: [M, 4]
    :return: [N, M]
    '''
    intersections = pairwise_intersection_area(boxes1, boxes2)  # [N, M]
    areas1 = boxes_area(boxes1)  # [N]
    areas2 = boxes_area(boxes2)  # [M]

    unions = tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections

    return tf.where(tf.equal(intersections, 0), tf.zeros_like(intersections), tf.truediv(intersections, unions))


def sample_proposal_boxes(boxes, gt_boxes_list, gt_labels_list, boxes_num_per_image,
                          foreground_thresh, foreground_ratio, batch_size):
    '''
    :param boxes: [batch_size, N, 4]
    :param gt_boxes_list: list[[None, 4]] len(list) = batch_size
    :param gt_labels_list: list[[None]] len(list) = batch_size
    '''
    sampled_boxes_list, sampled_labels_list, gt_pair_foreground_index_list = [], [], []

    boxes_list = tf.split(boxes, batch_size, axis=0)
    for i in range(batch_size):
        boxes = boxes_list[i]  # [N, 4]
        gt_boxes, gt_labels = gt_boxes_list[i], gt_labels_list[i]  # [M, 4], [M]

        iou = pairwise_iou(boxes, gt_boxes)  # [N, M]

        # add ground truth as proposals as well
        boxes = tf.concat([boxes, gt_boxes], axis=0)  # [N+M, 4]
        iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)  # [N+M, M]

        foreground_mask = tf.reduce_max(iou, axis=-1) > foreground_thresh
        foreground_index = tf.reshape(tf.where(foreground_mask), [-1])
        foreground_num = tf.minimum(tf.size(foreground_index), int(boxes_num_per_image * foreground_ratio))
        foreground_index = tf.random_shuffle(foreground_index)[:foreground_num]

        background_mask = tf.logical_not(foreground_mask)
        background_index = tf.reshape(tf.where(background_mask), [-1])
        background_num = tf.minimum(boxes_num_per_image - foreground_num, tf.size(background_index))
        background_index = tf.random_shuffle(background_index)[:background_num]

        gt_index_pair_foreground = tf.gather(tf.argmax(iou, axis=-1), foreground_index)

        all_indices = tf.concat([foreground_index, background_index], axis=0)
        sampled_boxes = tf.gather(boxes, all_indices)
        sampled_labels = tf.concat(
            [tf.gather(gt_labels, gt_index_pair_foreground), tf.zeros_like(background_index, dtype=tf.int64)], axis=0)

        sampled_boxes_list.append(tf.stop_gradient(sampled_boxes))  # list([len(foreground_index) + len(background_index), 4])
        sampled_labels_list.append(tf.stop_gradient(sampled_labels))  # list([len(foreground_index) + len(background_index)])
        gt_pair_foreground_index_list.append(tf.stop_gradient(gt_index_pair_foreground))  # list([len(foreground_index)])

    return sampled_boxes_list, sampled_labels_list, gt_pair_foreground_index_list  # len(list) = batch_size

