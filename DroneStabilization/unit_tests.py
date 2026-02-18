from datetime import datetime as dt
import os
import unittest
import shutil
from PIL import Image
import yaml
import numpy as np
import pandas as pd
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


if __name__ == '__main__':
    unittest.main()
