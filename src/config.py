import os


class ProjConfig():
    """
    Project configurations
    """
    project_dir = "/Users/guangmingliu/Work/cv_obj_detection_demo/"

    # data specifications
    train_image_path = os.path.join(project_dir, 'data/iSAID/val/truck/train')
    train_anno_path = os.path.join(project_dir, 'data/iSAID/val/truck/train_anno.json')
    train_data_name = 'isaid_truck_train'

    val_image_path = os.path.join(project_dir, 'data/iSAID/val/truck/val')
    val_anno_path = os.path.join(project_dir, 'data/iSAID/val/truck/val_anno.json')
    val_data_name = 'isaid_truck_val'

    test_image_path = val_image_path
    test_anno_path = val_anno_path
    test_data_name = val_data_name

    # model dir
    model_dir = os.path.join(project_dir, 'models')


    def __repr__(self):
        rep = f'Project directory: {self.project_dir}\
        Data names: {self.train_data_name} {self.val_data_name} {self.test_data_name}'
        return rep

