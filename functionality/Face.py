class Face:
    wait_till_delete = 10
    TAKE_AVERAGE_OF = 10

    def __init__(self, rio):
        self.done_calculating = False
        self.mask_detected = None
        self.mask_detections = []

        self.set_rio(rio)

    def set_rio(self, rio):
        self.rio = rio

    def track_mask_detections(self, mask_or_not):
        self.mask_detections.append(mask_or_not)
        if len(self.mask_detections) >= self.TAKE_AVERAGE_OF:
            self.average_mask_detections()
            self.done_calculating = True

    def average_mask_detections(self):
        if self.mask_detections.count("Mask") >= len(self.mask_detections) / 2:
            self.mask_detected = True
        elif self.mask_detections.count("No mask") > len(self.mask_detections) / 2:
            self.mask_detected = False
