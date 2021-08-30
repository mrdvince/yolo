class YoloLayer(nn.Module):
    def __init__(self, anchor_mask=None, num_classes=0, anchors=None, num_anchors=1):
        super(YoloLayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) / num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = 32
        self.seen = 0

    def forward(self, output, nms_thresh):
        self.thresh = nms_thresh
        masked_anchors = []

        for m in self.anchor_mask:
            masked_anchors += self.anchors[
                m * self.anchor_step : (m + 1) * self.anchor_step
            ]

        masked_anchors = [anchor / self.stride for anchor in masked_anchors]
        boxes = get_region_boxes(
            output.data,
            self.thresh,
            self.num_classes,
            masked_anchors,
            len(self.anchor_mask),
        )

        return boxes

