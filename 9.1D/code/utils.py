import os

def create_artifacts_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
