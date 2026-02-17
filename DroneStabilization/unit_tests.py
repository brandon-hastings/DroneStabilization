from datetime import datetime as dt
import os
import unittest
import shutil
from PIL import Image
import yaml
import numpy as np
import pandas as pd
import yolo_train
from pathlib import Path
from utils import yaml_utils, directory_utils, tk_popup_utils
from new_project import new_project


def assert_exist(path):
    if not os.path.isdir(path):
        raise AssertionError("Folder does not exist: %s" % str(path))

class NewProjectTest(unittest.TestCase):
    # also tests create and write config functions

    # set up temp directories
    def setUp(self):
        self.testing_folder = os.path.join(os.getcwd(), "testing")
        self.load_dir = os.path.join(os.getcwd(), "testing", "load_dir")
        self.test_dir = os.path.join(os.getcwd(), "testing", "test_dir")
        os.makedirs(self.load_dir)
        os.makedirs(self.test_dir)
        self.test_project_folder = "-".join(["test_exp", "test_person", dt.today().strftime('%Y-%m-%d')])
        self.test_config_path = os.path.join(self.test_dir, self.test_project_folder, "config.yaml")

    # tear down temp directories
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.load_dir)
        shutil.rmtree(self.testing_folder)
    
    def test_new_project_single_folder(self):
        # give directory name holding images as list
        test_folder_path = os.path.join(self.load_dir, "test_folder")
        os.mkdir(test_folder_path)
        # create CR2 raw image in test_folder_path
        open(os.path.join(test_folder_path, "image1.CR2"), 'x').close()
        # function call takes folder path as list
        test_images_folder = [test_folder_path]
        # assert test
        self.assertEqual(new_project(project="test_exp",
                                     experimenter="test_person",
                                     folder=test_images_folder,
                                     image_type=".CR2",
                                     working_directory=self.test_dir),
                         self.test_config_path)

        # check for copied images in project folder
        # check that copied images path exists
        path_to_check = os.path.join(self.test_dir, self.test_project_folder, "original_images", "test_folder")
        assert_exist(path_to_check)

    def test_new_project_multi_folder(self):
        # give directory name holding directories of images as list
        # give directory name holding images as list
        test_folder_path1 = os.path.join(self.load_dir, "test_folder1")
        test_folder_path2 = os.path.join(self.load_dir, "test_folder2")
        os.mkdir(test_folder_path1)
        os.mkdir(test_folder_path2)
        # create CR2 raw image in test_folder_path
        open(os.path.join(test_folder_path1, "image1.CR2"), 'x').close()
        open(os.path.join(test_folder_path2, "image1.CR2"), 'x').close()
        # function call takes folder path as list
        test_images_folder = [test_folder_path1, test_folder_path2]
        # assert test
        self.assertEqual(new_project(project="test_exp",
                                     experimenter="test_person",
                                     folder=test_images_folder,
                                     image_type=".CR2",
                                     working_directory=self.test_dir),
                         self.test_config_path)

        # check that copied images path exists
        path_to_check = os.path.join(self.test_dir, self.test_project_folder, "original_images", "test_folder1")
        assert_exist(path_to_check)


class YamlUtilsTest(unittest.TestCase):

    # set up temp directories
    def setUp(self):
        self.testing_folder = os.path.join(os.getcwd(), "testing")
        self.load_dir = os.path.join(os.getcwd(), "testing", "load_dir")
        self.test_dir = os.path.join(os.getcwd(), "testing", "test_dir")
        os.makedirs(self.load_dir)
        os.makedirs(self.test_dir)
        self.test_project_folder = "-".join(["test_exp", "test_person", dt.today().strftime('%Y-%m-%d')])
        self.test_config_path = os.path.join(self.test_dir, self.test_project_folder, "config.yaml")

    # tear down temp directories
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.load_dir)
        shutil.rmtree(self.testing_folder)

    def test_create_config(self):
        pass

    def test_read_config(self):
        pass

    def test_write_config(self):
        pass


class DirectoryUtilsTest(unittest.TestCase):

    # set up temp directories
    def setUp(self):
        self.testing_folder = os.path.join(os.getcwd(), "testing")
        self.load_dir = os.path.join(os.getcwd(), "testing", "load_dir")
        self.test_dir = os.path.join(os.getcwd(), "testing", "test_dir")
        os.makedirs(self.load_dir)
        os.makedirs(self.test_dir)
        self.project_folder_name = "-".join(["test_exp", "test_person", dt.today().strftime('%Y-%m-%d')])
        self.test_project_folder = os.path.join(self.test_dir, self.project_folder_name)
        os.mkdir(self.test_project_folder)
        self.test_config_path = os.path.join(self.test_project_folder, "config.yaml")

    # tear down temp directories
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.load_dir)
        shutil.rmtree(self.testing_folder)

    def test_make_folder(self):
        test_folder = os.path.join(self.test_project_folder, "folder1")
        self.assertEqual(directory_utils.make_folder(self.test_project_folder, "folder1"), test_folder)

        assert_exist(os.path.join(self.test_project_folder, "folder1"))

    def test_search_existing_directories(self):
        test_original_folder = os.path.join(self.test_project_folder, "original_images")
        os.mkdir(test_original_folder)
        test_config = {
            "project_path": self.test_project_folder,
            "image_folders": ["folder1", "folder2", "folder3"]
        }

        # correct test outcomes
        # Path object generated as save file
        test_new_path = Path(os.path.join(self.test_project_folder, "test_new"))
        # list of folders as Path objects to process still
        folders_to_process = [Path(os.path.join(test_original_folder, i)) for i in
                              test_config["image_folders"]]

        self.assertEqual(directory_utils.search_existing_directories(test_config, "test_new", "original_images"),
                         (folders_to_process, test_new_path))

        # assert that new save path was created
        assert_exist(test_new_path)

    def test_replace_path(self):
        test_given_path = os.path.join(self.test_dir, "test_folder", "folder1")
        replacement_folder = "replacement_folder"
        solution = os.path.join(self.test_dir, replacement_folder, "folder1")
        self.assertEqual(directory_utils.replace_path(test_given_path, find="test_folder", replace=replacement_folder),
                         solution)
        self.assertEqual(directory_utils.replace_path(Path(test_given_path), find="test_folder", replace=replacement_folder),
                         solution)

    # function not used in project
    def test_find_directories(self):
        test_folder_path = os.path.join(self.test_dir, "test_folder")
        os.mkdir(test_folder_path)
        os.mkdir(os.path.join(test_folder_path, "folder1"))
        os.mkdir(os.path.join(test_folder_path, "folder2"))
        open(os.path.join(test_folder_path, "image1.CR2"), 'x').close()

        pass

    def test_correct_path(self):
        func_input = self.test_dir + "/folder/"
        solution = self.test_dir + "/folder"
        self.assertEqual(directory_utils.correct_path(func_input), solution)


class TKPopupUtilsTest(unittest.TestCase):

    # set up temp directories
    def setUp(self):
        self.testing_folder = os.path.join(os.getcwd(), "testing")
        self.load_dir = os.path.join(os.getcwd(), "testing", "load_dir")
        self.test_dir = os.path.join(os.getcwd(), "testing", "test_dir")
        os.makedirs(self.load_dir)
        os.makedirs(self.test_dir)
        self.project_folder_name = "-".join(["test_exp", "test_person", dt.today().strftime('%Y-%m-%d')])
        self.test_project_folder = os.path.join(self.test_dir, self.project_folder_name)
        os.mkdir(self.test_project_folder)
        self.test_config_path = os.path.join(self.test_project_folder, "config.yaml")

    # tear down temp directories
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.load_dir)
        shutil.rmtree(self.testing_folder)


class Yolov5TrainTest(unittest.TestCase):

    # set up temp directories
    def setUp(self):
        self.testing_folder = os.path.join(os.getcwd(), "testing")
        self.load_dir = os.path.join(os.getcwd(), "testing", "load_dir")
        self.test_dir = os.path.join(os.getcwd(), "testing", "test_dir")
        os.makedirs(self.load_dir)
        os.makedirs(self.test_dir)
        self.project_folder_name = "-".join(["test_exp", "test_person", dt.today().strftime('%Y-%m-%d')])
        self.test_project_folder = os.path.join(self.test_dir, self.project_folder_name)
        os.mkdir(self.test_project_folder)
        self.test_config_path = os.path.join(self.test_project_folder, "config.yaml")

    # tear down temp directories
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.load_dir)
        shutil.rmtree(self.testing_folder)

    def test_preprocess_csv(self):
        image_paths = [os.path.join(self.load_dir, "images", i) for i in ["image1", "image2", "image3", "image4"]]
        coordinates = [(12, 23, 42, 54), (11, 22, 41, 53), (12, 23, 42, 54), (11, 22, 41, 53)]
        classes = ["A", "B", "A", "B"]
        test_csv = pd.DataFrame({"A": coordinates, "B": coordinates})
        test_csv.index.name = "paths"
        test_csv.index = image_paths
        test_csv.reset_index()
        test_classes = test_csv.columns.values.tolist()
        test_dict = test_csv.to_dict('index')
        # begin some dictionary gymnastics
        for key, value in test_dict.items():
            # overwrite original key with values as tuples
            test_dict[key] = [(cls, coord) for cls, coord in value.items()]

        self.assertEqual(yolo_train.preprocess_csv(test_csv),
                         (test_dict, test_classes))  # Tuples should match

    def test_format_yolo_training_data(self):
        test_img = np.ones((100, 100, 3), dtype="uint8")
        test_img = test_img * 255
        im = Image.fromarray(test_img)
        im.convert("RGB")
        test_img_path = os.path.join(self.load_dir, "image.png")
        im.save(test_img_path)
        test_dict = {test_img_path: {"A": (20, 20, 80, 80), "B": (10, 10, 90, 90)}}

        label_metadata_dict = {test_img_path: [("A", .5, .5, .6, .6), ("B", .5, .5, .8, .8)]}

        self.assertEqual(yolo_train.format_yolo_training_data(data_dict=test_dict),
                         label_metadata_dict) # Training data from list is not matching test dictionary

    def test_train_validation_split(self):
        image_paths = [os.path.join(self.load_dir, "images", i) for i in ["image1", "image2", "image3", "image4"]]
        test_split = 0.5
        test_label_dict = {}
        for i in range(len(image_paths)):
            test_label_dict[image_paths[i]] = [("A", .5, .5, .6, .6), ("B", .5, .5, .8, .8)]

        self.assertEqual(len(yolo_train.train_validation_split(label_metadata_dict=test_label_dict,
                                                           test_split=.5)),
                         2)

    def test_yolo_file_structuring(self):
        test_config = {"project_path": self.test_project_folder}
        test_config_path = self.test_config_path
        with open(test_config_path, "w") as file:
            yaml.dump(test_config, file, default_flow_style=False)
            os.makedirs(os.path.join(self.test_dir, "images"))
        image_paths = [os.path.join(self.test_dir, "images", i) for i in ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]]
        classes = ["A", "B"]
        train_dict = {}
        valid_dict = {}
        for i in image_paths:
            open(i, 'x').close()
        for i in range(0, 2):
            train_dict[image_paths[i]] = [("A", .5, .5, .6, .6), ("B", .5, .5, .8, .8)]
        for i in range(2, 4):
            valid_dict[image_paths[i]] = [("A", .5, .5, .6, .6), ("B", .5, .5, .8, .8)]
        print(train_dict)
        for key, value in valid_dict.items():
            print(os.path.basename(key).split('.')[0])
            for v in value:
                print(v)

        solution_config = os.path.join(self.test_project_folder, "bounding_boxes", "data.yaml")

        self.assertEqual(yolo_train.yolo_file_structuring(config_file=test_config_path,
                                                          train_set=train_dict,
                                                          validation_set=valid_dict,
                                                          classes=classes),
                         solution_config)

        print(os.listdir(os.path.join(self.test_project_folder, "bounding_boxes", "train", "images")))
        print(os.listdir(os.path.join(self.test_project_folder, "bounding_boxes", "train", "labels")))
        print(os.listdir(os.path.join(self.test_project_folder, "bounding_boxes", "valid", "images")))
        print(os.listdir(os.path.join(self.test_project_folder, "bounding_boxes", "valid", "labels")))

        assert_exist(os.path.join(self.test_project_folder, "bounding_boxes", "valid"))
        assert_exist(os.path.join(self.test_project_folder, "bounding_boxes", "test"))
        assert_exist(os.path.join(self.test_project_folder, "bounding_boxes", "train"))
        assert_exist(os.path.join(self.test_project_folder, "bounding_boxes", "valid", "images"))
        assert_exist(os.path.join(self.test_project_folder, "bounding_boxes", "valid", "labels"))
        assert(os.path.isfile(os.path.join(self.test_project_folder, "bounding_boxes", "valid", "labels", "image3_metadata.txt")))


if __name__ == '__main__':
    unittest.main()
