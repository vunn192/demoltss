from upload_app.helpers.encoder.color_encoder import color_encoder
import numpy as np

def fill(img, height, width, channel_num):
    img_filled = img.copy()
    height_pad = 0

    if height % 8 != 0:
        height_pad = 8 - (height % 8)
        filler = np.ones((height_pad, width, channel_num), dtype=np.uint8) * 128
        img_filled = np.concatenate([img, filler], axis = 0)

    if width % 8 != 0:
        filler = np.ones((height + height_pad, 8 - (width % 8), channel_num), dtype=np.uint8) * 128
        img_filled = np.concatenate([img_filled, filler], axis = 1)

    return img_filled


def jpeg_encoder(image_out_dir, img, height, width, quality):
    img = fill(img, height, width, 3)
    color_encoder(image_out_dir, img, height, width, quality)
