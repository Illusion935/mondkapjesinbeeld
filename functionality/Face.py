class Face:
    wait_till_delete = 10
    TAKE_AVERAGE_OF = 10

    def __init__(self, roi, pos_emoji=None, neg_emoji=None, roi_img=None):
        self.done_calculating = False
        self.mask_detected = None
        self.mask_detections = []
        self.positive_emoji_img = pos_emoji
        self.negative_emoji_img = neg_emoji
        self.new_face = True

        self.roi = roi
        self.roi_img = roi_img

    def count_mask_detections(self, mask_or_not):
        self.mask_detections.append(mask_or_not)
        if len(self.mask_detections) >= self.TAKE_AVERAGE_OF:
            self.average_mask_detections()
            self.done_calculating = True

    def average_mask_detections(self):
        if self.mask_detections.count("Mask") >= len(self.mask_detections) / 2:
            self.mask_detected = True
        elif self.mask_detections.count("No mask") > len(self.mask_detections) / 2:
            self.mask_detected = False
