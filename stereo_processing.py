import numpy as np
import cv2
import kornia as K
import torch
from kornia.feature import LoFTR

def find_outliers(data):

    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    up_thresh = Q3 + 1.5 * IQR
    under_thresh = Q1 - 1.5 * IQR
    indexes_up = np.where(data > up_thresh)[0]
    indexes_under = np.where(data < under_thresh)[0]

    return np.concatenate((indexes_under, indexes_up))

def add_intermediate_points(mkpts0, mkpts1):
  """
  Takes 3 near points (near by coordinates, not position in array) and places
  one more between them for both arrays.

  Args:
    mkpts0: Array of keypoints for image 1.
    mkpts1: Array of keypoints for image 2.

  Returns:
    Updated arrays mkpts0 and mkpts1 with added intermediate points.
  """

  distances = np.linalg.norm(mkpts0[:, None, :] - mkpts0[None, :, :], axis=2)
  near_indices = np.argwhere(distances < 10)  

  # Select 3 near points
  if len(near_indices) >= 3:
    idx1, idx2 = near_indices[0][:2]
    pt1 = mkpts0[idx1]
    pt2 = mkpts0[idx2]
    pt3 = mkpts0[near_indices[1][1]]

    # Calculate intermediate point
    new_pt = (pt1 + pt2 + pt3) / 3

    # Add the new point to mkpts0
    mkpts0 = np.vstack([mkpts0, new_pt])

    # Repeat for mkpts1
    pt1 = mkpts1[idx1]
    pt2 = mkpts1[idx2]
    pt3 = mkpts1[near_indices[1][1]]
    new_pt = (pt1 + pt2 + pt3) / 3
    mkpts1 = np.vstack([mkpts1, new_pt])

  return mkpts0, mkpts1

def loftr_feature_matching(img1, img2):
    device = K.utils.get_cuda_or_mps_device_if_available()
    # Преобразование изображений в тензоры и их нормализация
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    
    img1 = K.geometry.resize(img1, (512, 512), antialias=True)
    img2 = K.geometry.resize(img2, (512, 512), antialias=True)

    matcher = LoFTR(pretrained="indoor_new")
    input_dict = {
        "image0": K.color.rgb_to_grayscale(img1),
        "image1": K.color.rgb_to_grayscale(img2),
    }
    
    with torch.inference_mode():
        correspondences = matcher(input_dict)
    
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    
    return mkpts0, mkpts1