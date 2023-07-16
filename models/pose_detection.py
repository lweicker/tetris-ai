import json
import os
import sys

import numba as nb
import cv2
import PIL.Image
import torch
import torchvision.transforms as transforms
import numpy as np
import trt_pose
import trt_pose.models
from trt_pose.parse_objects import ParseObjects

from models.data_models import DetectionPoint, Pose

HUMAN_POSE_JSON_PATH = 'human_pose.json'
MODEL_PATH = 'resnet18_baseline_att_224x224_A_epoch_249.pth'


class PoseDetection:
    def __init__(self):
        topology, num_links, num_parts = self._load_topology()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(self.device)
        self.model = self._load_model(num_links, num_parts)
        self.parse_objects = ParseObjects(topology)
        self.image_size = 224
        self.mean = torch.Tensor([0.485, 0.456, 0.406])
        self.std = torch.Tensor([0.229, 0.224, 0.225])
        if self.device == "cuda":
            trt_model_path = "trt-model-fp16.pth"
            if not (os.path.isfile(trt_model_path)):
                self._convert_model_to_trt(trt_model_path)
                print("Model has been converted to TensorRT. Please rerun the application.")
                sys.exit(1)
            print("Loading TRT model.")
            self.model = self._load_trt_model(trt_model_path)
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
            self.model(torch.zeros((1, 3, self.image_size, self.image_size)).cuda())

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        frame = cv2.resize(image, (self.image_size, self.image_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = PIL.Image.fromarray(frame)
        frame = transforms.functional.to_tensor(frame).to(self.torch_device)
        frame.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return frame[None, ...]

    def predict_human_poses(self, image: np.ndarray) -> Pose:
        preprocessed_images = self._preprocess(image)
        with torch.no_grad():
            cmap, paf = self.model(preprocessed_images)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = self.parse_objects(cmap, paf)
        return deduce_hands_position_from_detection(image.shape[1],
                                                    image.shape[0],
                                                    (counts.numpy()), objects.numpy(), peaks.numpy())

    @staticmethod
    def _load_topology():
        with open(HUMAN_POSE_JSON_PATH, 'r') as f:
            human_pose = json.load(f)
        skeleton = human_pose['skeleton']
        key_points = human_pose['keypoints']
        length_skeleton = len(skeleton)
        topology = torch.zeros((length_skeleton, 4)).int()
        for k in range(length_skeleton):
            topology[k][0] = 2 * k
            topology[k][1] = 2 * k + 1
            topology[k][2] = skeleton[k][0] - 1
            topology[k][3] = skeleton[k][1] - 1
        return topology, length_skeleton, len(key_points)

    def _load_model(self, num_links: int, num_parts: int):
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links,
                                                      pretrained=False)
        if self.device == 'cpu':
            model = model.eval()
        else:
            model = model.cuda().eval()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        return model

    @staticmethod
    def _load_trt_model(path: str):
        from torch2trt import TRTModule
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(path))
        model_trt.cuda().eval()
        return model_trt

    def _convert_model_to_trt(self, output: str):
        print("Converting model to TensorRT. It may take a while.")
        import torch2trt
        data = torch.zeros((1, 3, self.image_size, self.image_size)).cuda()
        model_trt = torch2trt.torch2trt(self.model, [data], fp16_mode=True,
                                        max_workspace_size=1 << 25,
                                        max_batch_size=1)
        torch.save(model_trt.state_dict(), output)


@nb.jit(nopython=True)
def deduce_hands_position_from_detection(width_target, height_target,
                                         object_counts, objects,
                                         normalized_peaks) -> Pose:
    count = int(object_counts[0])
    left_hand = DetectionPoint((-1, -1))
    right_hand = DetectionPoint((-1, -1))
    for i in range(count):
        obj = objects[0][i]
        C = obj.shape[0]
        for j in range(C):
            k = int(obj[j])
            if k >= 0:
                peak = normalized_peaks[0][j][k]
                x = round(float(peak[1]) * width_target)
                y = round(float(peak[0]) * height_target)
                if j == 9:
                    left_hand = DetectionPoint((x, y))
                elif j == 10:
                    right_hand = DetectionPoint((x, y))

    return Pose(left_hand, right_hand)
