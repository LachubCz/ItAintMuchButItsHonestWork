class Image(object):
    def __init__(self, filename, gt_ellipse_center_x, gt_ellipse_center_y, gt_ellipse_majoraxis, 
                     gt_ellipse_minoraxis, gt_ellipse_angle, image_width, image_height, category):
        self.filename = filename
        self.gt_ellipse_center_x = gt_ellipse_center_x
        self.gt_ellipse_center_y = gt_ellipse_center_y
        self.gt_ellipse_majoraxis = gt_ellipse_majoraxis
        self.gt_ellipse_minoraxis = gt_ellipse_minoraxis
        self.gt_ellipse_angle = gt_ellipse_angle
        self.image_width = image_width
        self.image_height = image_height
        self.category = category
