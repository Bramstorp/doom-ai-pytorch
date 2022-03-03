import torch
import vizdoom
import skimage.transform
import numpy as np

from vizdoom import Mode


resolution = (30, 45)

def create_doom_env(config_file_path):
    game = vizdoom.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized")

    return game


def image_preprocessing(image):
    image = skimage.transform.resize(image, resolution)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

# Uses GPU if available
DEVICE = ""
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device('cpu')
