from typing import Tuple

import cv2
from numba.experimental import jitclass

NOT_ACTIVE_COLOR = (255, 0, 0)
ACTIVE_COLOR = (0, 255, 0)
TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
TEXT_SCALE = 1.2
TEXT_THICKNESS = 1
CIRCLE_THICKNESS = 4


@jitclass
class DetectionPoint:
    position: Tuple[int, int]
    x: int
    y: int

    def __init__(self, position: Tuple[int, int]):
        self.position = position
        self.x = position[0]
        self.y = position[1]

    def is_point_in(self, shape: "Circle") -> bool:
        return (self.x - shape.center.x) ** 2 + (self.y - shape.center.y) ** 2 < shape.radius ** 2

    def is_point_valid(self) -> bool:
        if self.x != -1 and self.y != -1:
            return True
        return False


@jitclass
class Circle:
    center: DetectionPoint
    radius: int

    def __init__(self, center: DetectionPoint, radius: int):
        self.center = center
        self.radius = radius


@jitclass
class Pose:
    left_hand: DetectionPoint
    right_hand: DetectionPoint

    def __init__(self, left_hand: DetectionPoint, right_hand: DetectionPoint):
        self.left_hand = left_hand
        self.right_hand = right_hand


@jitclass
class ActionZones:
    left: Circle
    right: Circle
    rotate: Circle
    down: Circle

    def __init__(self, left: Circle, right: Circle,
                 rotate: Circle, down: Circle):
        self.left = left
        self.right = right
        self.rotate = rotate
        self.down = down


@jitclass
class PreviousActions:
    left: bool
    right: bool
    rotate: bool
    down: bool
    action_zones: ActionZones
    dx_action: int
    rotate_action: bool
    anim_limit: int
    can_go_down: bool

    def __init__(self, left: bool, right: bool,
                 rotate: bool, down: bool, action_zones: ActionZones, anim_limit: int, dx_action: int = 0,
                 rotate_action: bool = False, can_go_down: bool = True):
        self.left = left
        self.right = right
        self.rotate = rotate
        self.down = down
        self.action_zones = action_zones
        self.anim_limit = anim_limit
        self.dx_action = dx_action
        self.rotate_action = rotate_action
        self.can_go_down = can_go_down

    def update_actions(self, detections: Pose) -> None:
        rotate_from_left = False
        rotate_from_right = False
        self.dx_action = 0
        self.rotate_action = False
        left_hand = detections.left_hand
        right_hand = detections.right_hand
        if left_hand.is_point_valid():
            if left_hand.is_point_in(self.action_zones.left):
                if not self.left:
                    self.dx_action = -1
                self.left = True
            elif left_hand.is_point_in(self.action_zones.rotate):
                if not self.rotate:
                    self.rotate_action = True
                self.rotate = True
                rotate_from_left = True
            elif left_hand.is_point_in(self.action_zones.down) and self.can_go_down:
                self.anim_limit = 100
            else:
                self.left = False
                rotate_from_left = False

        if right_hand.is_point_valid():
            if right_hand.is_point_in(self.action_zones.right):
                if not self.right:
                    self.dx_action = 1
                self.right = True
            elif right_hand.is_point_in(self.action_zones.rotate):
                if not self.rotate:
                    self.rotate_action = True
                self.rotate = True
                rotate_from_right = True
            elif right_hand.is_point_in(self.action_zones.down) and self.can_go_down:
                self.anim_limit = 100
            else:
                self.right = False
                rotate_from_right = False

        if right_hand.is_point_valid() and not right_hand.is_point_in(self.action_zones.down) \
                and left_hand.is_point_valid() and not left_hand.is_point_in(self.action_zones.down):
            self.can_go_down = True

        if self.anim_limit == 100:
            self.down = True
        if not rotate_from_right and not rotate_from_left:
            self.rotate = False
