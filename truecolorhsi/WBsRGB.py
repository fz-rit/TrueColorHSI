## White-balance model class
#
# Copyright (c) 2018-present, Mahmoud Afifi
# York University, Canada
# mafifi@eecs.yorku.ca | m.3afifi@gmail.com
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# All rights reserved.
#
# Please cite the following work if this program is used:
# Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown,
# "When color constancy goes wrong: Correcting improperly white-balanced
# images", CVPR 2019.
#
##########################################################################


import numpy as np
import numpy.matlib
import cv2
from skimage.feature import blob_log
from skimage.color import rgb2gray

class Classic_WB:
    """
    A class implementing Classic White Balance correction methods:
    - Gray World Assumption
    - White Patch (Max RGB) Assumption
    """
    def checkout_input_range(self, image: np.ndarray) -> bool:
        """
        Checks if the input image is in the range [0, 1].
        
        Args:
            image (np.ndarray): Input image as a NumPy array of shape (H, W, 3).
        
        Returns:
            bool: True if the image is in the range [0, 1], False otherwise.
        """
        if np.min(image) < 0 or np.max(image) > 1:
            raise ValueError(f"Input image of the wb algorithm must be in the range [0, 1]. Current range: [{np.min(image)}, {np.max(image)}]")

    
    def gray_world_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Gray World white balance to an image.
        
        The algorithm assumes that the average color in a scene is gray, and scales
        the red and blue channels based on the green channel's mean intensity.
        
        Args:
            image (np.ndarray): Input image as a NumPy array of shape (H, W, 3).
        
        Returns:
            np.ndarray: White-balanced image with pixel values normalized between [0, 1].
        """
        self.checkout_input_range(image)
        
        # Compute the mean of each channel
        mean_R = np.mean(image[:, :, 2])  # Red channel
        mean_G = np.mean(image[:, :, 1])  # Green channel
        mean_B = np.mean(image[:, :, 0])  # Blue channel
        
        # Compute scaling factors
        k_R = mean_G / mean_R
        k_B = mean_G / mean_B
        
        outImg = image.copy()
        # Apply scaling
        outImg[:, :, 2] = np.clip(image[:, :, 2] * k_R, 0, 1)
        outImg[:, :, 0] = np.clip(image[:, :, 0] * k_B, 0, 1)
        
        return outImg
    
    def white_patch_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Applies White Patch white balance to an image using blob detection to locate
        the most probable white patch.
        
        The algorithm detects bright regions in the image using blob detection,
        and scales the red and blue channels based on the brightest patch's intensity.
        
        Args:
            image (np.ndarray): Input image as a NumPy array of shape (H, W, 3).
        
        Returns:
            np.ndarray: White-balanced image with pixel values normalized between [0, 1].
        """
        self.checkout_input_range(image)
        
        # Convert image to grayscale for blob detection
        gray = rgb2gray(image)
        
        # Detect blobs using Laplacian of Gaussian (LoG)
        blobs = blob_log(gray, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.1)
        
        if len(blobs) > 0:
            # Find the brightest detected blob
            blobs = sorted(blobs, key=lambda b: gray[int(b[0]), int(b[1])], reverse=True)
            y, x, _ = blobs[0]  # Get the coordinates of the brightest blob
            max_R = image[int(y), int(x), 2]
            max_G = image[int(y), int(x), 1]
            max_B = image[int(y), int(x), 0]
        else:
            # Fallback to max channel values if no blobs are detected
            max_R = np.max(image[:, :, 2])
            max_G = np.max(image[:, :, 1])
            max_B = np.max(image[:, :, 0])
        
        # Compute scaling factors
        k_R = max_G / max_R
        k_B = max_G / max_B
        
        # Apply scaling
        image[:, :, 2] = np.clip(image[:, :, 2] * k_R, 0, 1)
        image[:, :, 0] = np.clip(image[:, :, 0] * k_B, 0, 1)
        
        return image



class WBsRGB:
  def __init__(self, gamut_mapping=2, upgraded=0):
    if upgraded == 1:
      self.features = np.load('models/features+.npy')  # encoded features
      self.mappingFuncs = np.load('models/mappingFuncs+.npy')  # correct funcs
      self.encoderWeights = np.load('models/encoderWeights+.npy')  # PCA matrix
      self.encoderBias = np.load('models/encoderBias+.npy')  # PCA bias
      self.K = 75  # K value for NN searching
    else:
      self.features = np.load('models/features.npy')  # encoded features
      self.mappingFuncs = np.load('models/mappingFuncs.npy')  # correction funcs
      self.encoderWeights = np.load('models/encoderWeights.npy')  # PCA matrix
      self.encoderBias = np.load('models/encoderBias.npy')  # PCA bias
      self.K = 25  # K value for nearest neighbor searching

    self.sigma = 0.25  # fall-off factor for KNN blending
    self.h = 60  # histogram bin width
    # our results reported with gamut_mapping=2, however gamut_mapping=1
    # gives more compelling results with over-saturated examples.
    self.gamut_mapping = gamut_mapping  # options: 1 scaling, 2 clipping

  def encode(self, hist):
    """ Generates a compacted feature of a given RGB-uv histogram tensor."""
    histR_reshaped = np.reshape(np.transpose(hist[:, :, 0]),
                                (1, int(hist.size / 3)), order="F")
    histG_reshaped = np.reshape(np.transpose(hist[:, :, 1]),
                                (1, int(hist.size / 3)), order="F")
    histB_reshaped = np.reshape(np.transpose(hist[:, :, 2]),
                                (1, int(hist.size / 3)), order="F")
    hist_reshaped = np.append(histR_reshaped,
                              [histG_reshaped, histB_reshaped])
    feature = np.dot(hist_reshaped - self.encoderBias.transpose(),
                     self.encoderWeights)
    return feature

  def rgb_uv_hist(self, I):
    """ Computes an RGB-uv histogram tensor. """
    sz = np.shape(I)  # get size of current image
    if sz[0] * sz[1] > 202500:  # resize if it is larger than 450*450
      factor = np.sqrt(202500 / (sz[0] * sz[1]))  # rescale factor
      newH = int(np.floor(sz[0] * factor))
      newW = int(np.floor(sz[1] * factor))
      I = cv2.resize(I, (newW, newH), interpolation=cv2.INTER_NEAREST)
    I_reshaped = I[(I > 0).all(axis=2)]
    eps = 6.4 / self.h
    hist = np.zeros((self.h, self.h, 3))  # histogram will be stored here
    Iy = np.linalg.norm(I_reshaped, axis=1)  # intensity vector
    for i in range(3):  # for each histogram layer, do
      r = []  # excluded channels will be stored here
      for j in range(3):  # for each color channel do
        if j != i:
          r.append(j)
      Iu = np.log(I_reshaped[:, i] / I_reshaped[:, r[1]])
      Iv = np.log(I_reshaped[:, i] / I_reshaped[:, r[0]])
      hist[:, :, i], _, _ = np.histogram2d(
        Iu, Iv, bins=self.h, range=((-3.2 - eps / 2, 3.2 - eps / 2),) * 2, weights=Iy)
      norm_ = hist[:, :, i].sum()

      if norm_ == 0:
        print(f"Warning: norm_ is zero for channel {i}, skipping normalization.")
        continue
      hist[:, :, i] = np.sqrt(hist[:, :, i] / norm_)  # (hist/norm)^(1/2)
    return hist

  def correctImage(self, I):
    """ White balance a given image I. """
    I = I[..., ::-1]  # convert from BGR to RGB
    I = im2double(I)  # convert to double
    # Convert I to float32 may speed up the process.
    feature = self.encode(self.rgb_uv_hist(I))
    # Do
    # ```python
    # feature_diff = self.features - feature
    # D_sq = np.einsum('ij,ij->i', feature_diff, feature_diff)[:, None]
    # ```
    D_sq = np.einsum(
      'ij, ij ->i', self.features, self.features)[:, None] + np.einsum(
      'ij, ij ->i', feature, feature) - 2 * self.features.dot(feature.T)

    # get smallest K distances
    idH = D_sq.argpartition(self.K, axis=0)[:self.K]
    mappingFuncs = np.squeeze(self.mappingFuncs[idH, :])
    dH = np.sqrt(
      np.take_along_axis(D_sq, idH, axis=0))
    weightsH = np.exp(-(np.power(dH, 2)) /
                      (2 * np.power(self.sigma, 2)))  # compute weights
    weightsH = weightsH / sum(weightsH)  # normalize blending weights
    mf = sum(np.matlib.repmat(weightsH, 1, 33) *
             mappingFuncs, 0)  # compute the mapping function
    mf = mf.reshape(11, 3, order="F")  # reshape it to be 9 * 3
    I_corr = self.colorCorrection(I, mf)  # apply it!
    return I_corr

  def colorCorrection(self, input, m):
    """ Applies a mapping function m to a given input image. """
    sz = np.shape(input)  # get size of input image
    I_reshaped = np.reshape(input, (int(input.size / 3), 3), order="F")
    kernel_out = kernelP(I_reshaped)
    out = np.dot(kernel_out, m)
    if self.gamut_mapping == 1:
      # scaling based on input image energy
      out = normScaling(I_reshaped, out)
    elif self.gamut_mapping == 2:
      # clip out-of-gamut pixels
      out = outOfGamutClipping(out)
    else:
      raise Exception('Wrong gamut_mapping value')
    # reshape output image back to the original image shape
    out = out.reshape(sz[0], sz[1], sz[2], order="F")
    out = out.astype('float32')[..., ::-1]  # convert from BGR to RGB
    return out


def normScaling(I, I_corr):
  """ Scales each pixel based on original image energy. """
  norm_I_corr = np.sqrt(np.sum(np.power(I_corr, 2), 1))
  inds = norm_I_corr != 0
  norm_I_corr = norm_I_corr[inds]
  norm_I = np.sqrt(np.sum(np.power(I[inds, :], 2), 1))
  I_corr[inds, :] = I_corr[inds, :] / np.tile(
    norm_I_corr[:, np.newaxis], 3) * np.tile(norm_I[:, np.newaxis], 3)
  return I_corr


def kernelP(rgb):
  """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric
          characterization based on polynomial modeling." Color Research &
          Application, 2001. """
  r, g, b = np.split(rgb, 3, axis=1)
  return np.concatenate(
    [rgb, r * g, r * b, g * b, rgb ** 2, r * g * b, np.ones_like(r)], axis=1)


def outOfGamutClipping(I):
  """ Clips out-of-gamut pixels. """
  I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
  I[I < 0] = 0  # any pixel is below 0, clip it to 0
  return I


def im2double(im):
  """ Returns a double image [0,1] of the uint8 im [0,255]. """
  return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
