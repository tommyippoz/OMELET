import os
import shutil
import time


def current_ms():
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


def clean_name(file, prequel):
    """
    Method to get clean name of a file
    :param file: the original file path
    :return: the filename with no path and extension
    """
    if prequel in file:
        file = file.replace(prequel, "")
    if '.' in file:
        file = file.split('.')[0]
    if file.startswith("/"):
        file = file[1:]
    return file


def get_full_class_name(class_obj):
    return class_obj.__module__ + "." + class_obj.__qualname__


def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
