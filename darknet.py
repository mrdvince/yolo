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


def get_region_boxes(
    output,
    conf_thresh,
    num_classes,
    anchors,
    num_anchors,
    only_objectness=1,
    validation=False,
):
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert output.size(1) == (5 + num_classes) * num_anchors
    print("########", output.shape)
    h = output.size(2)
    w = output.size(3)
    all_boxes = []
    output = (
        output.view(batch * num_anchors, 5 + num_classes, h * w)
        .transpose(0, 1)
        .contiguous()
        .view(5 + num_classes, batch * num_anchors * h * w)
    )
    grid_x = (
        torch.linspace(0, w - 1, w)
        .repeat(h, 1)
        .repeat(batch * num_anchors, 1, 1)
        .view(batch * num_anchors * h * w)
        .type_as(output)
    )  # cuda()
    grid_y = (
        torch.linspace(0, h - 1, h)
        .repeat(w, 1)
        .t()
        .repeat(batch * num_anchors, 1, 1)
        .view(batch * num_anchors * h * w)
        .type_as(output)
    )  # cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = (
        torch.Tensor(anchors)
        .view(num_anchors, anchor_step)
        .index_select(1, torch.LongTensor([0]))
    )
    anchor_h = (
        torch.Tensor(anchors)
        .view(num_anchors, anchor_step)
        .index_select(1, torch.LongTensor([1]))
    )
    anchor_w = (
        anchor_w.repeat(batch, 1)
        .repeat(1, 1, h * w)
        .view(batch * num_anchors * h * w)
        .type_as(output)
    )  # cuda()
    anchor_h = (
        anchor_h.repeat(batch, 1)
        .repeat(1, 1, h * w)
        .view(batch * num_anchors * h * w)
        .type_as(output)
    )  # cuda()
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h
    det_confs = torch.sigmoid(output[4])
    cls_confs = torch.nn.Softmax(dim=1)(
        output[5 : 5 + num_classes].transpose(0, 1)
    ).detach()
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))

    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        # fmt: off
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
                        # fmt: on
        all_boxes.append(boxes)
    return all_boxes
