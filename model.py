import tensorflow as tf

from extract_feature import extract_feature
from region_proposal_network import rpn_head, generate_anchors, generate_rpn_proposals, rpn_losses
from process_box import offsets_to_boxes, roi_align
from fast_rcnn import sample_fast_rcnn_targets


class Faster_RCNN:
    def __init__(self):
        # backbone
        self.scale_ratio = 1 / 16

        # rpn
        self.anchor_sizes = [16, 32, 64]
        self.anchor_ratios = [0.5, 1, 2]
        self.anchors_num = len(self.anchor_sizes) * len(self.anchor_ratios)

        self.positive_anchor_threshold = 0.7
        self.negative_anchor_threshold = 0.3

        self.train_pre_nms_topk = 12000  # topk before nms
        self.train_post_nms_topk = 2000  # topk after nms
        self.test_pre_nms_topk = 6000
        self.test_post_nms_topk = 1000

        self.proposal_nms_threshold = 0.7  # iou threshold of nms
        self.rpn_batch_per_image = 256

        # fast rcnn
        self.foreground_threshold=0.5

    def backbone(self, images, training):
        return extract_feature(images, training)  # [batch_size, images_height/16, images_width/16, 512]

    def rpn(self, feature_map, gt_boxes, gt_labels, batch_size, training):
        '''
        :param feature_map: [batch_size, h, w, c]
        :param gt_boxes: list[[None, 4]] len(list) = batch_size
        :param gt_labels: list[[None]] len(list) = batch_size
        :return:
        '''
        feature_shape = feature_map.get_shape.as_list()[1:3]  # h,w

        # pred_labels: [batch, h, w, anchors_num*1]
        # pred_offsets: [batch, h, w, anchors_num*4]
        pred_labels, pred_offsets = rpn_head(feature_map, self.anchors_num)

        # [batch_size, h, w, anchors_num * 4]
        anchors = generate_anchors(self.anchor_sizes, self.anchor_ratios, feature_shape, batch_size)

        # [batch_size, h, w, anchors_num, 4]
        pred_boxes = offsets_to_boxes(
            tf.reshape(pred_offsets, [-1, feature_shape[0], feature_shape[1], self.anchors_num, 4]),
            tf.reshape(anchors, [-1, feature_shape[0], feature_shape[1], self.anchors_num, 4]))

        # proposal_boxes: [batch_size, None, 4]
        # proposal_scores: [batch_size, None]
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            tf.reshape(pred_boxes, [-1, feature_shape[0] * feature_shape[1] * self.anchors_num, 4]),
            tf.reshape(pred_labels, [-1, feature_shape[0] * feature_shape[1] * self.anchors_num]),
            feature_shape,
            self.train_pre_nms_topk if training else self.test_pre_nms_topk,
            self.train_post_nms_topk if training else self.test_post_nms_topk,
            self.proposal_nms_threshold,
            batch_size)

        if training:
            losses = rpn_losses(gt_boxes, anchors, pred_offsets, pred_labels,
                                self.positive_anchor_threshold, self.negative_anchor_threshold, batch_size)
        else:
            losses = []

        return proposal_boxes, losses

    def roi_head(self, images, feature_map, proposal_boxes, gt_boxes, gt_labels, crop_size, training):
        '''
        :param images: [batch_size, image_h, image_w, image_c]
        :param feature_map: [batch_size, h, w, c]
        :param proposal_boxes: [batch_size, N, 4]
        :param gt_boxes: list[[None, 4]] len(list) = batch_size
        :param gt_labels: list[[None]] len(list) = batch_size
        :param crop_size: int   # the size of image crops
        :param training: bool
        :return:
        '''
