import cv2
import numpy as np

class GPUWarpAffine:
    def __init__(self, device_id=0):
        cv2.cuda.setDevice(device_id)
        self.device_id = device_id

    def warp(self, frame_cpu, M, outW, outH, pad_right=0, pad_bottom=0):
        cv2.cuda.setDevice(self.device_id)

        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame_cpu)

        gpu_M = cv2.cuda_GpuMat()
        gpu_M.upload(M.astype(np.float32))

        gpu_out = cv2.cuda.warpAffine(
            gpu_frame, gpu_M,
            (outW, outH),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        out = gpu_out.download()

        if pad_right or pad_bottom:
            out = cv2.copyMakeBorder(
                out, 0, pad_bottom, 0, pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )
        return out