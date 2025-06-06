# Credit: Murat Kirtay, The BioRobotics Inst./SSSA/
# Date: 02/11/2016

import cv2
import numpy as np


def derive_graym(impath):
    """The intensity value m is calculated as (r+g+b)/3, yet
    grayscale will do same operation!
    opencv uses default formula Y = 0.299 R + 0.587 G + 0.114 B
    """
    # return cv2.imread(impath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    return cv2.imread(impath, cv2.IMREAD_GRAYSCALE)


def derive_m(img, rimg):
    """Derive m (intensity) based on paper formula"""

    (rw, cl, ch) = img.shape
    for r in range(rw):
        for c in range(cl):
            rimg[r, c] = int(np.sum(img[r, c]) / 3.0)

    return rimg


def derive_saturation(img, rimg):
    """Derive saturation value for a pixel based on paper formula"""

    s_img = np.array(rimg)
    (r, c) = s_img.shape
    for ri in range(r):
        for ci in range(c):
            # opencv ==> b,g,r order
            s1 = np.clip(img[ri, ci][0] + img[ri, ci][2], 0, 255)
            s2 = 2 * img[ri, ci][1]
            if s1 >= s2:
                s_img[ri, ci] = 1.5 * np.clip((img[ri, ci][2] - rimg[ri, ci]), 0, 255)
            else:
                s_img[ri, ci] = 1.5 * np.clip((rimg[ri, ci] - img[ri, ci][0]), 0, 255)

    return s_img


def check_pixel_specularity(mimg, simg, m_max=0.6, s_max=0.4):
    """Check whether a pixel is part of specular region or not"""

    m_max = np.max(mimg) * m_max
    s_max = np.max(simg) * s_max

    (rw, cl) = simg.shape

    spec_mask = np.zeros((rw, cl), dtype=np.uint8)
    for r in range(rw):
        for c in range(cl):
            if mimg[r, c] >= m_max and simg[r, c] <= s_max:
                spec_mask[r, c] = 255

    return spec_mask


def enlarge_specularity(spec_mask):
    """Use sliding window technique to enlarge specularity
    simply move window over the image if specular pixel detected
    mark center pixel is specular
    win_size = 5x5, step_size = 1
    """

    win_size, step_size = (5, 5), 1
    enlarged_spec = np.array(spec_mask)
    for r in range(0, spec_mask.shape[0], step_size):
        for c in range(0, spec_mask.shape[1], step_size):
            # yield the current window
            win = spec_mask[r : r + win_size[1], c : c + win_size[0]]

            if win.shape[0] == win_size[0] and win.shape[1] == win_size[1]:
                if win[1, 1] != 0:
                    enlarged_spec[r : r + win_size[1], c : c + win_size[0]] = 255 * np.ones((5, 5), dtype=np.uint8)

    return enlarged_spec
