# affine_cpu.py
import cv2
import numpy as np

def estimate_affine_cpu(src_pts, dst_pts, ransac_thresh=3.0, partial=True):
    """
    src_pts: Nx2 moving points
    dst_pts: Nx2 reference points
    """
    if partial:
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thresh),
            maxIters=4000,
            refineIters=15,
            confidence=0.995,
        )
    else:
        M, inliers = cv2.estimateAffine2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thresh),
            maxIters=4000,
            refineIters=15,
            confidence=0.995,
        )
    return M, inliers