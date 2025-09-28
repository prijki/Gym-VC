import cv2
import mediapipe as mp
import numpy as np

# Inicialização do MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Função para calcular ângulo entre 3 pontos
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ab = a - b
    cb = c - b
    # Calculo do cosceno do angulo utilizando a formula de divisão do produto escalar.
    cos_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-8)
    # Garantindo que o resultado esteja entre 1 e -1
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # Converte o valor para angulo em graus
    angle = np.degrees(np.arccos(cos_angle))
    return angle


# Contador e estado
counter_agachamento = 0
counter_rosca = 0
stage = None  # "baixo" ou "cima"

# Captura da webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Processamento do frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Agachamento
            hip_lm = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            knee_lm = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            ankle_lm = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            ##Rosca
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # Checar visibilidade dos pontos
            if hip_lm.visibility > 0.7 and knee_lm.visibility > 0.7 and ankle_lm.visibility > 0.7:
                hip = [hip_lm.x, hip_lm.y]
                knee = [knee_lm.x, knee_lm.y]
                ankle = [ankle_lm.x, ankle_lm.y]

                # Calcular ângulo do joelho
                knee_angle = calculate_angle(hip, knee, ankle)

                # Contador de agachamento
                if knee_angle < 90:
                    stage = "baixo"
                if knee_angle > 160 and stage == "baixo":
                    stage = "cima"
                    counter_agachamento += 1

                # Mostrar ângulo na tela
                cv2.putText(image, f"Angulo joelho: {int(knee_angle)}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Checar visibilidade dos pontos
            if shoulder.visibility > 0.7 and elbow.visibility > 0.7 and wrist.visibility > 0.7:
                shoulder = [shoulder.x, shoulder.y]
                elbow = [elbow.x, elbow.y]
                wrist = [wrist.x, wrist.y]

                # Calcular ângulo do cotovelo
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                # Contador de rosca
                if elbow_angle < 40:
                    stage = "baixo"
                if elbow_angle > 160 and stage == "baixo":
                    stage = "cima"
                    counter_rosca += 1

                # Mostrar ângulo na tela
                cv2.putText(image, f"Angulo rosca: {int(elbow_angle)}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        except:
            pass

        # Desenhar landmarks e conexões
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Mostrar contador de repetições
        cv2.putText(image, f"Agachamentos: {counter_agachamento}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(image, f"Roscas Biceps: {counter_rosca}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar vídeo
        cv2.imshow("Detector de Agachamento", image)

        # Pressione 'q' para sair
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
