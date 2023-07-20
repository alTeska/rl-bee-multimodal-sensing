import os


def create_directory(path):
    # Create directories for trained model and log saving
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
