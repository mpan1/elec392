from utils.detection_receiver import DetectionReceiver
from picarx import Picarx
import time

car = Picarx()
detector = DetectionReceiver()

while True:
    detector.update()
    detections = detector.get_latest()

    if detections is None:
        car.stop()  # fail-safe
    else:
        objects = detections["objects"]
        if objects:
            # Example: steer away from detected object
            car.set_dir_servo_angle(-10)
        else:
            car.set_dir_servo_angle(0)

    time.sleep(0.05)