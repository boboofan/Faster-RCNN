import numpy as np
import tensorflow as tf

from process_box import clip_boxes, boxes_to_offsets
from fast_rcnn import pairwise_iou


def conv(inputs, output_dim, kernel_size=3, strides=1, padding='same', activation=None):
    return tf.layers.conv2d(inputs=inputs,
                            filters=output_dim,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            activation=activation,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))


def rpn_head(feature_map, anchors_num):
    '''
    :return: label_logits: [batch, h, w, anchors_num*1]
             box_logits: [batch, h, w, anchors_num*4]
    '''
    input_dim = feature_map.get_shape().as_list()[-1]
    head = conv(feature_map, input_dim, activation=tf.nn.relu)  # [batch, h, w, input_dim]

    label_logits = conv(head, anchors_num * 1, 1)  # [batch, h, w, anchors_num*1]
    box_logits = conv(head, anchors_num * 4, 1)  # [batch, h, w, anchors_num*4]

    return label_logits, box_logits


def generate_anchors(sizes, ratios, feature_shape, batch_size):
    anchors = []
    for size in sizes:
        for ratio in ratios:
            # w*h = area
            # h/w = ratio
            w = np.sqrt(size * size / ratio)
            h = w * ratio
            anchors.append([-w, -h, w, h])
    cell_anchors = np.asarray(anchors) * 0.5

    offset_x = np.arange(0, feature_shape[0])
    offset_y = np.arange(0, feature_shape[1])

    offset_x, offset_y = np.meshgrid(offset_x, offset_y)
    offset_x = offset_x.flatten()
    offset_y = offset_y.flatten()
    offsets = np.vstack([offset_x, offset_y, offset_x, offset_y]).transpose()

    # offsets_num = feature_height * feature_width
    offsets_num = offsets.shape[0]

    # cell_anchors_num=len(sizes)*len(ratios)
    cell_anchors_num = cell_anchors.shape[0]

    all_anchors = cell_anchors.reshape([1, cell_anchors_num, 4]) + \
                  offsets.reshape([1, offsets_num, 4]).transpose([1, 0, 2])
    all_anchors = all_anchors.reshape([feature_shape[0], feature_shape[1], cell_anchors_num * 4]).astype(np.float32)
    all_anchors = np.tile(all_anchors[np.newaxis], [batch_size, 1, 1, 1])

    return all_anchors  # [batch_size, feature_height, feature_width, anchors_num * 4]


def batch_gather(tensor, indices, batch_size):
    '''
    implement function as tf.gather for a batch
    :param tensor:  [batch_size, ...]
    :param indices: [batch_size, ...]
    :return: [batch_size, ...]
    '''
    tensor_list = tf.split(tensor, batch_size, axis=0)
    indices_list = tf.split(indices, batch_size, axis=0)

    gathered_tensor_list = [tf.gather(tensor_list[i], indices_list[i]) for i in range(batch_size)]

    return tf.concat(gathered_tensor_list, axis=0)


def generate_rpn_proposals(boxes, scores, feature_shape, pre_nms_topk, post_nms_topk, proposal_nms_thresh, batch_size):
    '''
    :param boxes: [batch_size, N, 4]
    :param scores: [batch_size, N]
    :param feature_shape: h, w
    '''

    topk = tf.minimum(pre_nms_topk, tf.shape(scores)[-1])

    # topk_scores, topk_indices: [batch_size, top_k]
    topk_scores, topk_indices = tf.nn.top_k(scores, k=topk, sorted=False)

    # topk_boxes: [batch_size, top_k, 4]
    topk_boxes = batch_gather(boxes, topk_indices, batch_size)
    topk_boxes = clip_boxes(topk_boxes, feature_shape)

    # h, w: [batch_size, topk, 1]
    min_xs, min_ys, max_xs, max_ys = tf.split(topk_boxes, 4, axis=-1)
    w = max_xs - min_xs
    h = max_ys - min_ys

    vaild_mask = tf.concat([w > 0, h > 0], axis=-1)
    vaild_mask = tf.reduce_all(vaild_mask, axis=-1)  # [batch_size, topk]
    topk_vaild_boxes = tf.boolean_mask(topk_boxes, vaild_mask)  # [batch_size, valid_num, 4]
    topk_vaild_scores = tf.boolean_mask(topk_scores, vaild_mask)  # [batch_size, valid_num]

    # non max suppression
    x1y1, x2y2 = tf.split(topk_vaild_boxes, 2, axis=-1)  # [batch_size, valid_num, 2]
    y1x1 = tf.reverse(x1y1, axis=-1)
    y2x2 = tf.reverse(x2y2, axis=-1)

    y1x1_list = tf.split(y1x1, batch_size, axis=0)
    y2x2_list = tf.split(y2x2, batch_size, axis=0)
    topk_vaild_scores_list = tf.split(topk_vaild_scores, batch_size, 0)
    nms_indices_list = [tf.image.non_max_suppression(tf.concat([y1x1_list[i], y2x2_list[i]], axis=-1),
                                                     topk_vaild_scores_list[i],
                                                     max_output_size=post_nms_topk,
                                                     iou_threshold=proposal_nms_thresh) for i in range(batch_size)]
    nms_indices = tf.concat(nms_indices_list, axis=0)

    proposal_boxes = batch_gather(topk_vaild_boxes, nms_indices, batch_size)
    proposal_scores = batch_gather(topk_vaild_scores, nms_indices, batch_size)

    return tf.stop_gradient(proposal_boxes, name='boxes'), tf.stop_gradient(proposal_scores, name='scores')


def reg_loss(anchors, gt_boxes, pred_offsets):
    '''
    :param anchors: [N, 4]
    :param gt_boxes: [N, 4]
    :param pred_offsets: [N, 4]
    '''
    gt_offsets = boxes_to_offsets(gt_boxes, anchors)

    # 0.5 * x ^ 2                                       if | x | <= delta
    # 0.5 * delta ^ 2 + delta * (| x | - delta)         if | x | > delta
    delta = 1 / 9
    loss = tf.losses.huber_loss(gt_offsets, pred_offsets, delta=delta, reduction=tf.losses.Reduction.SUM) / delta

    return loss


def cls_loss(gt_labels, pred_labels):
    '''
    :param gt_labels: [N]
    :param pred_labels: [N]
    '''
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_labels, logits=pred_labels))

    return loss


def rpn_losses(gt_boxes_list, anchors, pred_offsets, pred_labels,
               positive_anchor_threshold, negative_anchor_threshold, batch_size):
    '''
    :param gt_boxes_list: list[[None, 4]] len(list) = batch_size
    :param anchors: [batch_size, h, w, 4*anchors_num]
    :param pred_offsets: [batch_size, h, w, 4*anchors_num]
    :param pred_labels: [batch_size, h, w, 1*anchors_num]
    '''

    anchors_list = tf.split(tf.reshape(anchors, [batch_size, -1, 4]), batch_size, axis=0)
    pred_offsets_list = tf.split(tf.reshape(pred_offsets, [batch_size, -1, 4]), batch_size, axis=0)
    pred_labels_list = tf.split(tf.reshape(pred_labels, [batch_size, -1]), batch_size, axis=0)

    cls_losses, reg_losses = [], []

    for i in range(len(anchors_list)):
        anchors = anchors_list[i]
        gt_boxes = gt_boxes_list[i]
        pred_offsets, pred_labels = pred_offsets_list[i], pred_labels_list[i]

        iou = pairwise_iou(anchors, gt_boxes)  # [anchors_num, gt_boxes_num]

        # [anchors_num] bool
        positive_mask = tf.reduce_max(iou, axis=1) > positive_anchor_threshold
        negative_mask = tf.reduce_max(iou, axis=1) < negative_anchor_threshold

        # reg loss
        positive_anchors = tf.boolean_mask(anchors, positive_mask)  # [p, 4]

        gt_boxes_pair_positive_index = tf.argmax(tf.boolean_mask(iou, positive_mask), axis=1)
        gt_boxes_pair_positive = tf.gather(gt_boxes, gt_boxes_pair_positive_index)  # [p, 4]

        pred_offsets_pair_positive = tf.boolean_mask(pred_offsets, positive_mask)

        reg_losses.append(reg_loss(positive_anchors, gt_boxes_pair_positive, pred_offsets_pair_positive))

        # cls loss
        vaild_mask = positive_mask | negative_mask
        vaild_pred_labels = tf.boolean_mask(pred_labels, vaild_mask)

        zeros = tf.zeros_like(pred_labels, dtype=tf.int32)
        gt_labels = tf.where(positive_mask, tf.ones_like(zeros, dtype=tf.int32), zeros)
        vaild_gt_labels = tf.boolean_mask(gt_labels, vaild_mask)

        cls_losses.append(cls_loss(vaild_gt_labels, vaild_pred_labels))

    return [tf.reduce_mean(cls_losses), tf.reduce_mean(reg_losses)]
