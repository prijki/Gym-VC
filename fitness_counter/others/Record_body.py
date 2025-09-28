import cv2
import mediapipe as mp

# Inicializa o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Abre a câmera
cap = cv2.VideoCapture(0)  # 0 = webcam padrão

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Não conseguiu acessar a câmera.")
        break

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Captura landmarks como dicionário
        landmarks = {}
        for id, lm in enumerate(result.pose_landmarks.landmark):
            landmarks[id] = {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}

    # Mostra janela
    cv2.imshow("Gravador de Coreografia", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('s'):  
        # Salva a pose com timestamp
        coreografia.append({
            "time": time.time(),  # tempo absoluto (pode ser ajustado depois para sincronizar com música)
            "pose": landmarks
        })
        print("Pose salva!")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Exporta para JSON
with open("coreografia.json", "w") as f:
    json.dump(coreografia, f, indent=4)

print("Coreografia salva em coreografia.json")