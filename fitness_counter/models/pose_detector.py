import mediapipe as mp

class PoseDetector:
    def __init__(self, detection_conf=0.7, tracking_conf=0.7):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process(self, image):
        return self.pose.process(image)

    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
        )