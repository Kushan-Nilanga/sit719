import os

def create_artifacts_dir(postfix):
    artifacts_dir = os.path.join(os.getcwd(), 'artefacts')
    if not os.path.exists(artifacts_dir):
        os.mkdir(artifacts_dir)
    artifacts_dir = os.path.join(artifacts_dir, postfix)
    if not os.path.exists(artifacts_dir):
        os.mkdir(artifacts_dir)
    return artifacts_dir
