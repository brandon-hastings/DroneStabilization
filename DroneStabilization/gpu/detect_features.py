# detect_features.py
import cv2
import numpy as np

class GPUFeatureExtractor:
    """
    GPU SIFT/ORB feature extractor + GPU BFMatcher.
    All operations run on a *specific* GPU device_id.
    """

    def __init__(self, device_id=0, nfeatures=2000, use_orb_fallback=True):
        cv2.cuda.setDevice(device_id)
        self.device_id = device_id
        self.nfeatures = nfeatures

        # Try SIFT → fallback to ORB
        if hasattr(cv2.cuda, "SIFT_create"):
            self.det = cv2.cuda.SIFT_create(nfeatures=nfeatures)
            self.norm = cv2.NORM_L2
            self.name = "CUDA-SIFT"
        elif use_orb_fallback:
            self.det = cv2.cuda.ORB_create(nfeatures=nfeatures)
            self.norm = cv2.NORM_HAMMING
            self.name = "CUDA-ORB"
        else:
            raise RuntimeError("No CUDA SIFT available and ORB fallback disabled.")

        self.matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(self.norm)

    def extract(self, roi_cpu):
        """
        Input: roi_cpu (uint8 grayscale)
        Output: (keypoints_cpu, descriptors_cpu)
        """
        cv2.cuda.setDevice(self.device_id)

        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(roi_cpu)

        gpu_kp, gpu_desc = self.det.detectAndComputeAsync(gpu_mat, None)
        kp = self.det.convert(gpu_kp)
        desc = gpu_desc.download() if gpu_desc is not None else None
        return kp, desc

    def match(self, des_ref, des_mov, k=2):
        """
        GPU KNN matching → CPU list of matches
        """
        if des_ref is None or des_mov is None:
            return []

        cv2.cuda.setDevice(self.device_id)

        d_ref = cv2.cuda_GpuMat()
        d_mov = cv2.cuda_GpuMat()
        d_ref.upload(des_ref)
        d_mov.upload(des_mov)

        gpu_matches = self.matcher.knnMatch(d_ref, d_mov, k=k)
        return gpu_matches
