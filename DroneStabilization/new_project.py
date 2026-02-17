from datetime import datetime as dt
from pathlib import Path
import shutil
import os
import utils
import utils.directory_utils
import utils.yaml_utils


def new_project(project, experimenter, folder, image_type="", working_directory=None):

    months_3letter = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    date = dt.today()
    month = months_3letter[date.month]
    day = date.day
    d = str(month[0:3] + str(day))
    date = dt.today().strftime("%Y-%m-%d")
    if working_directory is None:
        working_directory = "."
    wd = Path(working_directory).resolve()
    project_name = "{pn}-{exp}-{date}".format(pn=project, exp=experimenter, date=date)
    project_path = wd / project_name
    os.makedirs(project_path)

    if len(folder) == 1:
        destination = project_path / "original_images"
        os.makedirs(destination)
        subfolder = os.path.join(destination, os.path.basename(folder[0]))
        os.makedirs(subfolder)
        for file_name in os.listdir(folder[0]):
            if file_name.lower().endswith(image_type):
                # print(file_name)
                # construct full file path
                source = Path(folder[0]) / file_name
                print(source)
                print(os.path.join(destination, os.path.basename(folder[0]), os.path.basename(source)))
                # copy only files
                if os.path.isfile(source):
                    shutil.copy(source, subfolder)
    if len(folder) > 1:
        destination = project_path / "original_images"
        for directory in folder:
            shutil.copytree(directory, destination / os.path.basename(directory))
    elif len(folder) == 0:
        print("No folders containing images passed to project file")

    cfg_file, ruamelFile = utils.yaml_utils.create_config()

    # common parameters:
    cfg_file["Task"] = project
    cfg_file["scorer"] = experimenter
    cfg_file["image_folders"] = folder
    cfg_file["project_path"] = str(project_path)
    cfg_file["date"] = d
    cfg_file["image_type"] = image_type

    projconfigfile = os.path.join(str(project_path), "config.yaml")
    # Write dictionary to yaml  config file
    utils.yaml_utils.write_config(projconfigfile, cfg_file)

    print("New Project created")
    return projconfigfile
