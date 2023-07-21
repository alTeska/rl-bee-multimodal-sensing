import os
import yaml
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def create_directory(path):
    """
    Create a directory at the given path if it does not already exist.

    Parameters:
        path (str): The directory path to create.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_config(config, output_path):
    """
    Save the configuration dictionary to a YAML file.

    Parameters:
        config (dict): The configuration dictionary to save.
        output_path (str): The path to the output directory where the config file will be saved.

    Returns:
        None
    """
    with open(os.path.join(output_path, "config.yaml"), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def display_video(frames, framerate=30):
    """
    Generate a video from a sequence of frames and display it.

    Parameters:
        frames (ndarray): Array of shape (n_frames, height, width, 3).
        framerate (int): Frame rate in units of Hz. Defaults to 30.

    Returns:
        IPython.display.DisplayObject: The display object containing the video.
    """
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False
    )
    return HTML(anim.to_html5_video())


def set_device():
    """
    Set the device to CUDA if available, otherwise to CPU.

    Returns:
        str: The device name (e.g., "cuda" or "cpu").
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(
            "WARNING: For this notebook to perform best, "
            "if possible, in the menu under `Runtime` -> "
            "`Change runtime type.`  select `GPU` "
        )
    else:
        print("GPU is enabled in this notebook.")

    return device


def retrieve_latest_model_path(path):
    """
    Retrieve the path of the latest model saved during training.

    Parameters:
        path (str): The directory path to search for the model files.

    Returns:
        str: The path of the latest model file (ending with ".zip").
    """
    max_number = float("-inf")
    max_filename = ""

    # Loop over the files in the folder
    for filename in os.listdir(path):
        if filename.endswith(".zip"):
            number = int(filename.split(".")[0])

            if number > max_number:
                max_number = number
                max_filename = filename

    return max_filename
