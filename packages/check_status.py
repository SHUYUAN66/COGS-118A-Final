import os

def check_directory(lst):
    for i in lst:
        CHECK_FOLDER = os.path.isdir(i)
        if not CHECK_FOLDER:
            os.makedirs(i)
        else:
            return
