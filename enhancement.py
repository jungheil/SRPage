import time

import numpy as np

# import PIL.Image as Image
import scipy as sp
from numba import jit


def box(img, r):
    """O(1) box filter
    img - >= 2d image
    r   - radius of box filter
    """
    (rows, cols) = img.shape[:2]
    imDst = np.zeros_like(img)

    tile = [1] * img.ndim
    tile[0] = r
    imCum = np.cumsum(img, 0)
    imDst[0 : r + 1, :, ...] = imCum[r : 2 * r + 1, :, ...]
    imDst[r + 1 : rows - r, :, ...] = (
        imCum[2 * r + 1 : rows, :, ...] - imCum[0 : rows - 2 * r - 1, :, ...]
    )
    imDst[rows - r : rows, :, ...] = (
        np.tile(imCum[rows - 1 : rows, :, ...], tile)
        - imCum[rows - 2 * r - 1 : rows - r - 1, :, ...]
    )

    tile = [1] * img.ndim
    tile[1] = r
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0 : r + 1, ...] = imCum[:, r : 2 * r + 1, ...]
    imDst[:, r + 1 : cols - r, ...] = (
        imCum[:, 2 * r + 1 : cols, ...] - imCum[:, 0 : cols - 2 * r - 1, ...]
    )
    imDst[:, cols - r : cols, ...] = (
        np.tile(imCum[:, cols - 1 : cols, ...], tile)
        - imCum[:, cols - 2 * r - 1 : cols - r - 1, ...]
    )

    return imDst


def _gf_color(I, p, r, eps, s=None):
    """Color guided filter
    I - guide image (rgb)
    p - filtering input (single channel)
    r - window radius
    eps - regularization (roughly, variance of non-edge noise)
    s - subsampling factor for fast guided filter
    """
    fullI = I
    fullP = p
    if s is not None:
        I = sp.ndimage.zoom(fullI, [1 / s, 1 / s, 1], order=1)
        p = sp.ndimage.zoom(fullP, [1 / s, 1 / s], order=1)
        r = round(r / s)

    h, w = p.shape[:2]
    N = box(np.ones((h, w)), r)

    mI_r = box(I[:, :, 0], r) / N
    mI_g = box(I[:, :, 1], r) / N
    mI_b = box(I[:, :, 2], r) / N

    mP = box(p, r) / N

    # mean of I * p
    mIp_r = box(I[:, :, 0] * p, r) / N
    mIp_g = box(I[:, :, 1] * p, r) / N
    mIp_b = box(I[:, :, 2] * p, r) / N

    # per-patch covariance of (I, p)
    covIp_r = mIp_r - mI_r * mP
    covIp_g = mIp_g - mI_g * mP
    covIp_b = mIp_b - mI_b * mP

    # symmetric covariance matrix of I in each patch:
    #       rr rg rb
    #       rg gg gb
    #       rb gb bb
    var_I_rr = box(I[:, :, 0] * I[:, :, 0], r) / N - mI_r * mI_r
    var_I_rg = box(I[:, :, 0] * I[:, :, 1], r) / N - mI_r * mI_g
    var_I_rb = box(I[:, :, 0] * I[:, :, 2], r) / N - mI_r * mI_b

    var_I_gg = box(I[:, :, 1] * I[:, :, 1], r) / N - mI_g * mI_g
    var_I_gb = box(I[:, :, 1] * I[:, :, 2], r) / N - mI_g * mI_b

    var_I_bb = box(I[:, :, 2] * I[:, :, 2], r) / N - mI_b * mI_b

    a = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            sig = np.array(
                [
                    [var_I_rr[i, j], var_I_rg[i, j], var_I_rb[i, j]],
                    [var_I_rg[i, j], var_I_gg[i, j], var_I_gb[i, j]],
                    [var_I_rb[i, j], var_I_gb[i, j], var_I_bb[i, j]],
                ]
            )
            covIp = np.array([covIp_r[i, j], covIp_g[i, j], covIp_b[i, j]])
            a[i, j, :] = np.linalg.solve(sig + eps * np.eye(3), covIp)

    b = mP - a[:, :, 0] * mI_r - a[:, :, 1] * mI_g - a[:, :, 2] * mI_b

    meanA = box(a, r) / N[..., np.newaxis]
    meanB = box(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, [s, s, 1], order=1)
        meanB = sp.ndimage.zoom(meanB, [s, s], order=1)

    q = np.sum(meanA * fullI, axis=2) + meanB

    return q


def _gf_gray(I, p, r, eps, s=None):
    """grayscale (fast) guided filter
    I - guide image (1 channel)
    p - filter input (1 channel)
    r - window raidus
    eps - regularization (roughly, allowable variance of non-edge noise)
    s - subsampling factor for fast guided filter
    """
    if s is not None:
        Isub = sp.ndimage.zoom(I, 1 / s, order=1)
        Psub = sp.ndimage.zoom(p, 1 / s, order=1)
        r = round(r / s)
    else:
        Isub = I
        Psub = p

    (rows, cols) = Isub.shape

    N = box(np.ones([rows, cols]), r)

    meanI = box(Isub, r) / N
    meanP = box(Psub, r) / N
    corrI = box(Isub * Isub, r) / N
    corrIp = box(Isub * Psub, r) / N
    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = box(a, r) / N
    meanB = box(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, s, order=1)
        meanB = sp.ndimage.zoom(meanB, s, order=1)

    q = meanA * I + meanB
    return q


def _gf_colorgray(I, p, r, eps, s=None):
    """automatically choose color or gray guided filter based on I's shape"""
    if I.ndim == 2 or I.shape[2] == 1:
        return _gf_gray(I, p, r, eps, s)
    elif I.ndim == 3 and I.shape[2] == 3:
        return _gf_color(I, p, r, eps, s)
    else:
        print("Invalid guide dimensions:", I.shape)


def guided_filter(I, p, r, eps, s=None):
    """run a guided filter per-channel on filtering input p
    I - guide image (1 or 3 channel)
    p - filter input (n channel)
    r - window raidus
    eps - regularization (roughly, allowable variance of non-edge noise)
    s - subsampling factor for fast guided filter
    """
    if p.ndim == 2:
        p3 = p[:, :, np.newaxis]

    out = np.zeros_like(p3)
    for ch in range(p3.shape[2]):
        out[:, :, ch] = _gf_colorgray(I, p3[:, :, ch], r, eps, s)
    return np.squeeze(out) if p.ndim == 2 else out


# def test_gf():
#     import imageio
#     cat = imageio.imread('cat.bmp').astype(np.float32) / 255
#     tulips = imageio.imread('tulips.bmp').astype(np.float32) / 255

#     r = 8
#     eps = 0.05

#     cat_smoothed = guided_filter(cat, cat, r, eps)
#     cat_smoothed_s4 = guided_filter(cat, cat, r, eps, s=4)

#     imageio.imwrite('cat_smoothed.png', cat_smoothed)
#     imageio.imwrite('cat_smoothed_s4.png', cat_smoothed_s4)

#     tulips_smoothed4s = np.zeros_like(tulips)
#     for i in range(3):
#         tulips_smoothed4s[:,:,i] = guided_filter(tulips, tulips[:,:,i], r, eps, s=4)
#     imageio.imwrite('tulips_smoothed4s.png', tulips_smoothed4s)

#     tulips_smoothed = np.zeros_like(tulips)
#     for i in range(3):
#         tulips_smoothed[:,:,i] = guided_filter(tulips, tulips[:,:,i], r, eps)
#     imageio.imwrite('tulips_smoothed.png', tulips_smoothed)


class HazeRemoval(object):
    def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
        pass

    # def open_image(self, img_path):
    #     img = Image.open(img_path)
    #     self.src = np.array(img).astype(np.double)/255.
    #     # self.gray = np.array(img.convert('L'))
    #     self.rows, self.cols, _ = self.src.shape
    #     self.dark = np.zeros((self.rows, self.cols), dtype=np.double)
    #     self.Alight = np.zeros((3), dtype=np.double)
    #     self.tran = np.zeros((self.rows, self.cols), dtype=np.double)
    #     self.dst = np.zeros_like(self.src, dtype=np.double)

    @jit
    def get_dark_channel(self, radius=7):
        print("Starting to compute dark channel prior...")
        start = time.time()
        tmp = self.src.min(axis=2)
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0, i - radius)
                rmax = min(i + radius, self.rows - 1)
                cmin = max(0, j - radius)
                cmax = min(j + radius, self.cols - 1)
                self.dark[i, j] = tmp[rmin : rmax + 1, cmin : cmax + 1].min()
        print("time:", time.time() - start)

    def get_air_light(self):
        print("Starting to compute air light prior...")
        start = time.time()
        flat = self.dark.flatten()
        flat.sort()
        num = int(self.rows * self.cols * 0.001)
        threshold = flat[-num]
        tmp = self.src[self.dark >= threshold]
        tmp.sort(axis=0)
        self.Alight = tmp[-num:, :].mean(axis=0)
        # print(self.Alight)
        print("time:", time.time() - start)

    @jit
    def get_transmission(self, radius=7, omega=0.95):
        print("Starting to compute transmission...")
        start = time.time()
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0, i - radius)
                rmax = min(i + radius, self.rows - 1)
                cmin = max(0, j - radius)
                cmax = min(j + radius, self.cols - 1)
                pixel = (self.src[rmin : rmax + 1, cmin : cmax + 1] / self.Alight).min()
                self.tran[i, j] = 1.0 - omega * pixel
        print("time:", time.time() - start)

    def guided_filter(self, r=60, eps=0.001):
        print("Starting to compute guided filter trainsmission...")
        start = time.time()
        self.gtran = guided_filter(self.src, self.tran, r, eps)
        print("time:", time.time() - start)

    def recover(self, t0=0.1):
        print("Starting recovering...")
        start = time.time()
        self.gtran[self.gtran < t0] = t0
        t = self.gtran.reshape(*self.gtran.shape, 1).repeat(3, axis=2)
        # import ipdb; ipdb.set_trace()
        self.dst = (self.src.astype(np.double) - self.Alight) / t + self.Alight
        self.dst *= 255
        self.dst[self.dst > 255] = 255
        self.dst[self.dst < 0] = 0
        self.dst = self.dst.astype(np.uint8)
        print("time:", time.time() - start)

    # def show(self):
    #     import cv2
    #     cv2.imwrite("img/src.jpg", (self.src*255).astype(np.uint8)[:,:,(2,1,0)])
    #     cv2.imwrite("img/dark.jpg", (self.dark*255).astype(np.uint8))
    #     cv2.imwrite("img/tran.jpg", (self.tran*255).astype(np.uint8))
    #     cv2.imwrite("img/gtran.jpg", (self.gtran*255).astype(np.uint8))
    #     cv2.imwrite("img/dst.jpg", self.dst[:,:,(2,1,0)])

    #     io.imsave("test.jpg", self.dst)

    def get(self, img):
        self.src = np.array(img).astype(np.double) / 255.0
        self.rows, self.cols, _ = self.src.shape
        self.dark = np.zeros((self.rows, self.cols), dtype=np.double)
        self.Alight = np.zeros((3), dtype=np.double)
        self.tran = np.zeros((self.rows, self.cols), dtype=np.double)
        self.dst = np.zeros_like(self.src, dtype=np.double)

        self.get_dark_channel()
        self.get_air_light()
        self.get_transmission()
        self.guided_filter()
        self.recover()

        return self.dst.copy()
