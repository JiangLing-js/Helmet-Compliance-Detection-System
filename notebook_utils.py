#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import threading
import time
import urllib.parse
from os import PathLike
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from openvino.runtime import Core, get_version
from IPython.display import HTML, Image, display


# ## Files
#
# Load an image, download a file, download an IR model, and create a progress bar to show download progress.

# In[ ]:


def load_image(path: str) -> np.ndarray:
    """
    Loads an image from `path` and returns it as BGR numpy array. `path`
    should point to an image file, either a local filename or a url. The image is
    not stored to the filesystem. Use the `download_file` function to download and
    store an image.

    :param path: Local path name or URL to image.
    :return: image as BGR numpy array
    """
    import cv2
    import requests

    if path.startswith("http"):
        # Set User-Agent to Mozilla because some websites block
        # requests with User-Agent Python
        response = requests.get(path, headers={"User-Agent": "Mozilla/5.0"})
        array = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(array, -1)  # Loads the image as BGR
    else:
        image = cv2.imread(path)
    return image


def download_file(
    url: PathLike,
    filename: PathLike = None,
    directory: PathLike = None,
    show_progress: bool = True,
    silent: bool = False,
    timeout: int = 10,
) -> PathLike:
    """
    Download a file from a url and save it to the local filesystem. The file is saved to the
    current directory by default, or to `directory` if specified. If a filename is not given,
    the filename of the URL will be used.

    :param url: URL that points to the file to download
    :param filename: Name of the local file to save. Should point to the name of the file only,
                     not the full path. If None the filename from the url will be used
    :param directory: Directory to save the file to. Will be created if it doesn't exist
                      If None the file will be saved to the current working directory
    :param show_progress: If True, show an TQDM ProgressBar
    :param silent: If True, do not print a message if the file already exists
    :param timeout: Number of seconds before cancelling the connection attempt
    :return: path to downloaded file
    """
    from tqdm.notebook import tqdm_notebook
    import requests

    filename = filename or Path(urllib.parse.urlparse(url).path).name
    chunk_size = 16384  # make chunks bigger so that not too many updates are triggered for Jupyter front-end

    filename = Path(filename)
    if len(filename.parts) > 1:
        raise ValueError(
            "`filename` should refer to the name of the file, excluding the directory. "
            "Use the `directory` parameter to specify a target directory for the downloaded file."
        )

    # create the directory if it does not exist, and add the directory to the filename
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / Path(filename)

    try:
        response = requests.get(url=url, headers={"User-agent": "Mozilla/5.0"}, stream=True)
        response.raise_for_status()
    except (
        requests.exceptions.HTTPError
    ) as error:  # For error associated with not-200 codes. Will output something like: "404 Client Error: Not Found for url: {url}"
        raise Exception(error) from None
    except requests.exceptions.Timeout:
        raise Exception(
            "Connection timed out. If you access the internet through a proxy server, please "
            "make sure the proxy is set in the shell from where you launched Jupyter."
        ) from None
    except requests.exceptions.RequestException as error:
        raise Exception(f"File downloading failed with error: {error}") from None

    # download the file if it does not exist, or if it exists with an incorrect file size
    filesize = int(response.headers.get("Content-length", 0))
    if not filename.exists() or (os.stat(filename).st_size != filesize):
        with tqdm_notebook(
            total=filesize,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=str(filename),
            disable=not show_progress,
        ) as progress_bar:
            with open(filename, "wb") as file_object:
                for chunk in response.iter_content(chunk_size):
                    file_object.write(chunk)
                    progress_bar.update(len(chunk))
                    progress_bar.refresh()
    else:
        if not silent:
            print(f"'{filename}' already exists.")

    response.close()

    return filename.resolve()


def download_ir_model(model_xml_url: str, destination_folder: PathLike = None) -> PathLike:
    """
    Download IR model from `model_xml_url`. Downloads model xml and bin file; the weights file is
    assumed to exist at the same location and name as model_xml_url with a ".bin" extension.

    :param model_xml_url: URL to model xml file to download
    :param destination_folder: Directory where downloaded model xml and bin are saved. If None, model
                               files are saved to the current directory
    :return: path to downloaded xml model file
    """
    model_bin_url = model_xml_url[:-4] + ".bin"
    model_xml_path = download_file(model_xml_url, directory=destination_folder, show_progress=False)
    download_file(model_bin_url, directory=destination_folder)
    return model_xml_path


# ## Images

# ### Convert Pixel Data
#
# Normalize image pixel values between 0 and 1, and convert images to RGB and BGR.

# In[ ]:


def normalize_minmax(data):
    """
    Normalizes the values in `data` between 0 and 1
    """
    if data.max() == data.min():
        raise ValueError("Normalization is not possible because all elements of" f"`data` have the same value: {data.max()}.")
    return (data - data.min()) / (data.max() - data.min())


def to_rgb(image_data: np.ndarray) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    import cv2

    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)


def to_bgr(image_data: np.ndarray) -> np.ndarray:
    """
    Convert image_data from RGB to BGR
    """
    import cv2

    return cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)


# ## Videos

# ### Video Player
#
# Custom video player to fulfill FPS requirements. You can set target FPS and output size, flip the video horizontally or skip first N frames.

# In[ ]:


class VideoPlayer:
    """
    Custom video player to fulfill FPS requirements. You can set target FPS and output size,
    flip the video horizontally or skip first N frames.

    :param source: Video source. It could be either camera device or video file.
    :param size: Output frame size.
    :param flip: Flip source horizontally.
    :param fps: Target FPS.
    :param skip_first_frames: Skip first N frames.
    """

    def __init__(self, source, size=None, flip=False, fps=None, skip_first_frames=0):
        import cv2

        self.cv2 = cv2  # This is done to access the package in class methods
        self.__cap = cv2.VideoCapture(source)
        if not self.__cap.isOpened():
            raise RuntimeError(f"Cannot open {'camera' if isinstance(source, int) else ''} {source}")
        # skip first N frames
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)
        # fps of input file
        self.__input_fps = self.__cap.get(cv2.CAP_PROP_FPS)
        if self.__input_fps <= 0:
            self.__input_fps = 60
        # target fps given by user
        self.__output_fps = fps if fps is not None else self.__input_fps
        self.__flip = flip
        self.__size = None
        self.__interpolation = None
        if size is not None:
            self.__size = size
            # AREA better for shrinking, LINEAR better for enlarging
            self.__interpolation = cv2.INTER_AREA if size[0] < self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH) else cv2.INTER_LINEAR
        # first frame
        _, self.__frame = self.__cap.read()
        self.__lock = threading.Lock()
        self.__thread = None
        self.__stop = False

    """
    Start playing.
    """

    def start(self):
        self.__stop = False
        self.__thread = threading.Thread(target=self.__run, daemon=True)
        self.__thread.start()

    """
    Stop playing and release resources.
    """

    def stop(self):
        self.__stop = True
        if self.__thread is not None:
            self.__thread.join()
        self.__cap.release()

    def __run(self):
        prev_time = 0
        while not self.__stop:
            t1 = time.time()
            ret, frame = self.__cap.read()
            if not ret:
                break

            # fulfill target fps
            if 1 / self.__output_fps < time.time() - prev_time:
                prev_time = time.time()
                # replace by current frame
                with self.__lock:
                    self.__frame = frame

            t2 = time.time()
            # time to wait [s] to fulfill input fps
            wait_time = 1 / self.__input_fps - (t2 - t1)
            # wait until
            time.sleep(max(0, wait_time))

        self.__frame = None

    """
    Get current frame.
    """

    def next(self):
        import cv2

        with self.__lock:
            if self.__frame is None:
                return None
            # need to copy frame, because can be cached and reused if fps is low
            frame = self.__frame.copy()
        if self.__size is not None:
            frame = self.cv2.resize(frame, self.__size, interpolation=self.__interpolation)
        if self.__flip:
            frame = self.cv2.flip(frame, 1)
        return frame


# ## Visualization

# ### Segmentation
#
# Define a SegmentationMap NamedTuple that keeps the labels and colormap for a segmentation project/dataset. Create CityScapesSegmentation and BinarySegmentation SegmentationMaps. Create a function to convert a segmentation map to an RGB image with a colormap, and to show the segmenta