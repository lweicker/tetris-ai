import numpy as np
import cv2

from models.data_models import ActionZones, Pose, PreviousActions

NOT_ACTIVE_COLOR = (255, 0, 0)
ACTIVE_COLOR = (0, 255, 0)
TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
TEXT_SCALE = 1.2
TEXT_THICKNESS = 1
CIRCLE_THICKNESS = 4
ACTIONS = ["left", "right", "rotate", "down"]


def add_circle_in_image(pose: Pose, image: np.ndarray) -> None:
    for var in dir(pose):
        if not var.startswith('_'):
            value = getattr(pose, var)
            if value:
                cv2.circle(image, value.position, 3, ACTIVE_COLOR, CIRCLE_THICKNESS)


def draw_zones_and_status(action_zones: ActionZones, image: np.ndarray, current_position_status: PreviousActions) -> None:
    for var in ACTIONS:
        value = getattr(action_zones, var)
        color = NOT_ACTIVE_COLOR
        if getattr(current_position_status, var):
            color = ACTIVE_COLOR
        cv2.circle(image, value.center.position, value.radius, color, CIRCLE_THICKNESS)
