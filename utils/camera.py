import threading
import cv2


class Camera:
    def __init__(self, src: int):
        self.cam = cv2.VideoCapture(src)
        _, self.img_handle = self.cam.read()
        self.thread = threading.Thread(target=self.get_image)
        self.thread.start()

    def get_image(self):
        while True:
            _, self.img_handle = self.cam.read()
            if self.img_handle is None:
                break

    def read(self):
        return self.img_handle

    def stop(self):
        self.cam.release()
