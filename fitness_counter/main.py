import cv2
from models.pose_detector import PoseDetector
from exercises.squat import SquatCounter

detector = PoseDetector()
squat = SquatCounter()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        angle, count= squat.update(results.pose_landmarks.landmark)
        detector.draw_landmarks(image, results)
        cv2.putText(image, f"Agachamentos: {count}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

        if angle is not None:
            cv2.putText(image, f"Angulo: {int(angle)}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        else:
            # Opcional: mostrar um placeholder
            cv2.putText(image, "Angulo: --", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        # cv2.putText(image, f"Angulo: {int(angle)}", (10, 130),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)

    cv2.imshow("Fitness Counter", image)
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
