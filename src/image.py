class Image(object):
    def __init__(self, image, processed_image, ground_truths, processed_ground_truths, filename, gt_ellipse_center_x, gt_ellipse_center_y, 
                 gt_ellipse_majoraxis, gt_ellipse_minoraxis, gt_ellipse_angle, image_width, 
                 image_height, category):
        self.image = image
        self.processed_image = processed_image
        self.ground_truths = ground_truths
        self.processed_ground_truths = processed_ground_truths
        self.filename = filename
        if gt_ellipse_center_x == '':
            self.ellipse = 0
        else:
            self.ellipse = 1
            self.gt_ellipse_center_x = float(gt_ellipse_center_x)
            self.gt_ellipse_center_y = float(gt_ellipse_center_y)
            self.gt_ellipse_majoraxis = float(gt_ellipse_majoraxis)
            self.gt_ellipse_minoraxis = float(gt_ellipse_minoraxis)
            self.gt_ellipse_angle = float(gt_ellipse_angle)

        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.category = int(category)
