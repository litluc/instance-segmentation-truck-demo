import os


class ProjConfig():
    """
    Project configurations
    """
    project_dir = "/Users/guangmingliu/Work/cv_obj_detection_demo/"
    train_image_path = os.path.join(project_dir, 'data/iSAID/val/truck/train')
    train_anno_path = os.path.join(project_dir, 'data/iSAID/val/truck/train_anno.json')
    val_image_path = os.path.join(project_dir, 'data/iSAID/val/truck/val')
    val_anno_path = os.path.join(project_dir, 'data/iSAID/val/truck/val_anno.json')

    def __repr__(self):
        rep = f'Project directory: {self.project_dir}'
        return rep

