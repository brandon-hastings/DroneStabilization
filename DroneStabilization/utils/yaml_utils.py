import os
import os.path
import yaml
from ruamel.yaml import YAML
from pathlib import Path

'''functions for creating and saving yaml config files (Mostly taken from DeepLabCut)'''


def create_config():
    yaml_str = """\
    # Project definitions (do not edit)
        Task:
        scorer:
        date:
        \n
    # Project path (change when moving around)
        project_path:
        image_folders:
        \n
    # Image type of raw images
        image_type:
        \n
    # Number of images to label
        n_images_label: 20
        \n
    # Bounding box object classes
        bbox_classes:
        - object1
        - object2
    # Bounding box training parameters
        init_lr: 1e-4
        num_epochs: 20
        batch_size: 32
        \n

    # Kepoint label classes
        keypoint_classes:
        - point1
        - point2
        """

    ruamelFile = YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return cfg_file, ruamelFile


def read_config(configname):
    """
    Reads structured config file defining a project.
    """
    ruamelFile = YAML()
    path = Path(configname)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = ruamelFile.load(f)
                curr_dir = os.path.dirname(configname)
                if cfg["project_path"] != curr_dir:
                    cfg["project_path"] = curr_dir
                    write_config(configname, cfg)
        except Exception as err:
            if len(err.args) > 2:
                if (
                        err.args[2]
                        == "could not determine a constructor for the tag '!!python/tuple'"
                ):
                    with open(path, "r") as ymlfile:
                        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                        write_config(configname, cfg)
                else:
                    raise

    else:
        raise FileNotFoundError(
            "Config file is not found. Please make sure that the file exists and/or that you passed the path of the config file correctly!"
        )
    return cfg


def write_config(configname, cfg):
    """
    Write structured config file.
    """
    with open(configname, "w+") as cf:
        cfg_file, ruamelFile = create_config()
        for key in cfg.keys():
            cfg_file[key] = cfg[key]
        ruamelFile.dump(cfg_file, cf)

