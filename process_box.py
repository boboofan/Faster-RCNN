import tensorflow as tf


def clip_boxes(boxes, window):
    '''
    :param boxes: min_x, min_y, max_x, max_y
    :param window: h, w
    '''
    boxes = tf.maximum(boxes, 0.0)

    max = tf.reverse(window, [0])  # w, h
    max = tf.tile(max, [2])  # w,h,w,h
    boxes = tf.minimum(boxes, tf.cast(max, tf.float32))

    return boxes


def boxes_to_offsets(boxes, anchors):
    '''
    :param boxes: [..., 4] min_x, min_y, max_x, max_y
    :param anchors: [..., 4] min_x, min_y, max_x, max_y
    :return: [..., 4] dx, dy, dw, dh
    '''
    boxes_min_x, boxes_min_y, boxes_max_x, boxes_max_y = tf.split(boxes, 4, axis=-1)
    boxes_width = boxes_max_x - boxes_min_x
    boxes_height = boxes_max_y - boxes_min_y
    boxes_center_x = boxes_min_x + boxes_width * 0.5
    boxes_center_y = boxes_min_y + boxes_height * 0.5

    anchors_min_x, anchors_min_y, anchors_max_x, anchors_max_y = tf.split(anchors, 4, axis=-1)
    anchors_width = anchors_max_x - anchors_min_x
    anchors_height = anchors_max_y - anchors_min_y
    anchors_center_x = anchors_min_x + anchors_width * 0.5
    anchors_center_y = anchors_min_y + anchors_height * 0.5

    dx = (boxes_center_x - anchors_center_x) / anchors_width
    dy = (boxes_center_y - anchors_center_y) / anchors_height
    dw = tf.log(boxes_width / anchors_width)
    dh = tf.log(boxes_height / anchors_height)

    return tf.concat([dx, dy, dw, dh], axis=-1)


def offsets_to_boxes(offsets, anchors):
    '''
    :param offsets: [..., 4] dx, dy, dw, dh
    :param anchors: [..., 4] min_x, min_y, max_x, max_y
    :return: [..., 4] min_x, min_y, max_x, max_y
    '''
    dx, dy, dw, dh = tf.split(offsets, 4, axis=-1)

    anchors_min_x, anchors_min_y, anchors_max_x, anchors_max_y = tf.split(anchors, 4, axis=-1)
    anchors_width = anchors_max_x - anchors_min_x
    anchors_height = anchors_max_y - anchors_min_y
    anchors_center_x = anchors_min_x + anchors_width * 0.5
    anchors_center_y = anchors_min_y + anchors_height * 0.5

    boxes_width = tf.exp(dw) * anchors_width
    boxes_height = tf.exp(dh) * anchors_height
    boxes_center_x = dx * anchors_width + anchors_center_x
    boxes_center_y = dy * anchors_height + anchors_center_y

    boxes_min_x = boxes_center_x - boxes_width * 0.5
    boxes_min_y = boxes_center_y - boxes_height * 0.5
    boxes_max_x = boxes_center_x + boxes_width * 0.5
    boxes_max_y = boxes_center_y + boxes_height * 0.5

    return tf.concat([boxes_min_x, boxes_min_y, boxes_max_x, boxes_max_y], axis=-1)


def roi_align(feature_map, boxes, box_ind, crop_size):
    '''
    :param featuremap: [batch_size, h, w, c]
    :param boxes: [N, 4]    # min_x, min_y, max_x, max_y
    :param box_ind: [N]     # it means the i-th box belongs to the box_ind[i]-th image
    :param crop_size: crop_height, crop_width
    :return: [N, crop_height, crop_width, c]
    '''

    # # TF's crop_and_resize produces zeros on border
    # if pad_border:
    #     # this can be quite slow
    #     image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
    #     boxes = boxes + 1

    feature_height, feature_width = feature_map.get_shape.as_list()[1:3]  # h,w

    # map value of x, y to [0, 1]
    boxes = tf.stop_gradient(boxes)
    min_x, min_y, max_x, max_y = tf.split(boxes, 4, axis=-1)
    min_x = min_x / (feature_width - 1)
    min_y = min_y / (feature_height - 1)
    max_x = max_x / (feature_width - 1)
    max_y = max_y / (feature_height - 1)
    boxes = tf.concat([min_y, min_x, max_y, max_x], axis=0)

    crops = tf.image.crop_and_resize(feature_map, boxes, box_ind, [crop_size[0] * 2, crop_size[1] * 2])
    avg_pool = tf.layers.average_pooling2d(crops, 2, 2, padding='same')

    return avg_pool
