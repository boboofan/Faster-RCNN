from easydict import EasyDict

cfg = EasyDict()
cfg.DATA = EasyDict()
cfg.RPN = EasyDict()

# data
cfg.DATA.IMAGE_SHAPE = [228, 228, 3]  # height, width, channel

# anchors
cfg.RPN.ANCHOR_STRIDE = 16
cfg.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
cfg.RPN.ANCHOR_RATIOS = (0.5, 1, 2)
cfg.RPN.MAX_SIZE = 1333
cfg.RPN.ANCHORS_NUM = len(cfg.RPN.ANCHOR_SIZES) * len(cfg.RPN.ANCHOR_RATIOS)

cfg.RPN.POSITIVE_ANCHOR_THRESH = 0.7
cfg.RPN.NEGATIVE_ANCHOR_THRESH = 0.3

cfg.RPN.TRAIN_PRE_NMS_TOPK = 12000
cfg.RPN.TRAIN_POST_NMS_TOPK = 2000
cfg.RPN.TEST_PRE_NMS_TOPK = 6000
cfg.RPN.TEST_POST_NMS_TOPK = 1000

cfg.RPN.MIN_SIZE = 0
cfg.RPN.PROPOSAL_NMS_THRESH = 0.7

cfg.RPN.BATCH_PER_IM = 256  # total (across FPN levels) number of anchors that are marked valid
