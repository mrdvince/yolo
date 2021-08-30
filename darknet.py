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


class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = (
            x.view(B, C, H, 1, W, 1)
            .expand(B, C, H, stride, W, stride)
            .contiguous()
            .view(B, C, H * stride, W * stride)
        )
        return x


# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

def parse_config(cfgfile):
    blocks = []
    fp = open(cfgfile, "r")
    block = None
    line = fp.readline()
    while line != "":
        line = line.rstrip()
        if line == "" or line[0] == "#":
            line = fp.readline()
            continue
        elif line[0] == "[":
            if block:
                blocks.append(block)
            block = dict()
            block["type"] = line.lstrip("[").rstrip("]")
            # set default value
            if block["type"] == "convolutional":
                block["batch_normalize"] = 0
        else:
            key, value = line.split("=")
            key = key.strip()
            if key == "type":
                key = "_type"
            value = value.strip()
            block[key] = value
        line = fp.readline()
    if block:
        blocks.append(block)
    fp.close()
    return blocks


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


