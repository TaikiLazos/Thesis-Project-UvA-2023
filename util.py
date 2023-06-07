import os


def make_folder(path):
    """
    If folder does not exist then make one.
    """
    if not os.path.exists(path):
        os.makedirs(path)
