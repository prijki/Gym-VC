# from utils.angle import calculate_angle_3d
# import mediapipe as mp
# import cv2

# class SquatCounter:
#     def __init__(self,
#                  visibility_thresh=0.25,
#                  down_flex_thresh=60.0,   # ajuste: quanto é considerado "descida" (flexão mínima)
#                  up_flex_thresh=30.0,
#                  min_consec_frames=3):    # ajuste: flexão abaixo disso = "em pé"
#         self.counter = 0
#         self.stage = "up"  # assume em pé no início
#         self.pose = mp.solutions.pose.PoseLandmark
#         self.visibility_thresh = visibility_thresh
#         self.down_flex_thresh = down_flex_thresh
#         self.up_flex_thresh = up_flex_thresh
#         self.min_consec_frames = min_consec_frames
#         self._down_frames = 0
#         self._up_frames = 0

#     def _raw_angle_for(self, landmarks, side):
#         hip   = landmarks[getattr(self.pose, f"{side}_HIP").value]
#         knee  = landmarks[getattr(self.pose, f"{side}_KNEE").value]
#         ankle = landmarks[getattr(self.pose, f"{side}_ANKLE").value]

#         # checa visibilidade
#         if (hip.visibility  < self.visibility_thresh or
#             knee.visibility < self.visibility_thresh or
#             ankle.visibility < self.visibility_thresh):
#             return None

#         a = [hip.x, hip.y, hip.z]
#         b = [knee.x, knee.y, knee.z]
#         c = [ankle.x, ankle.y, ankle.z]

#         return calculate_angle_3d(a, b, c)

#     def _flexion_for(self, landmarks, side):
#         """
#         Retorna a 'flexão' do joelho (quanto ele dobrou), em graus.
#         Faz flexion = 180 - ang_ext. Se não houver dados válidos retorna None.
#         """
#         raw = self._raw_angle_for(landmarks, side)
#         if raw is None:
#             return None
#         flexion = 180.0 - raw
#         return flexion

#     def update(self, landmarks):
#         # """
#         # Retorna (angle_display, counter)
#         # - angle_display: valor (em graus) para exibir (uso: max das flexões válidas)
#         # - counter: total de repetições
#         # """
#         # r_f = self._flexion_for(landmarks, "RIGHT")
#         # l_f = self._flexion_for(landmarks, "LEFT")

#         # flexions = [f for f in (r_f, l_f) if f is not None]
#         # if not flexions:
#         #     return None, self.counter

#         # # usamos o maior valor de flexão do frame para decidir (se UMA perna dobrou)
#         # current_flex = max(flexions)


#         # print("Contagem ",self.counter)
#         # print("current_flex: ", current_flex)


#         # # transição: up -> down quando a flexão excede down_flex_thresh
#         # if current_flex > self.down_flex_thresh and self.stage == "up":
#         #     print("Estado atual : ", self.stage)
#         #     self.stage = "down"

#         # # transição: down -> up quando a flexão cai abaixo de up_flex_thresh
#         # if current_flex < self.up_flex_thresh and self.stage == "down":
#         #     print("Estado atual : ", self.stage)
#         #     self.stage = "up"
#         #     self.counter += 1

#         # # retornamos a flexão usada para display (e a contagem)
#         # return current_flex, self.counter
#           # calcula flexões por lado
#         r_f = self._flexion_for(landmarks, "RIGHT")
#         l_f = self._flexion_for(landmarks, "LEFT")
#         flexions = [f for f in (r_f, l_f) if f is not None]
#         if not flexions:
#             # sem dados válidos: reset pequenos estados mas mantém counter
#             self._down_frames = 0
#             self._up_frames = 0
#             return None, self.counter

#         current_flex = max(flexions)  # usamos maior flexão do frame
#         # snapshot do estado anterior — evita testar transições não relevantes
#         prev_stage = self.stage

#         # --- lógica com debounce e sem duas transições no mesmo frame ---
#         if prev_stage == "up":
#             # estamos em pé, só avaliamos ir para down
#             if current_flex > self.down_flex_thresh:
#                 self._down_frames += 1
#             else:
#                 self._down_frames = 0

#             if self._down_frames >= self.min_consec_frames:
#                 self.stage = "down"
#                 self._down_frames = 0
#                 # opcional: print de debug
#                 print("TRANSIÇÃO: up -> down (confirmada)")

#         elif prev_stage == "down":
#             # estamos em down, só avaliamos ir para up
#             if current_flex < self.up_flex_thresh:
#                 self._up_frames += 1
#             else:
#                 self._up_frames = 0

#             if self._up_frames >= self.min_consec_frames:
#                 self.stage = "up"
#                 self.counter += 1
#                 self._up_frames = 0
#                 print("TRANSIÇÃO: down -> up (contada)")

#         # debug opcional para ver o que está acontecendo
#         print(f"Estado anterior: {prev_stage} | Estado atual: {self.stage} | current_flex: {current_flex:.2f} | Reps: {self.counter}")

#         return current_flex, self.counter


# squat.py
import mediapipe as mp
from utils.angle import calculate_angle_3d

class SquatCounter:
    def __init__(self, down_thresh=90, up_thresh=25, min_frames=5):
        self.counter = 0
        self.stage = "up"
        self.pose = mp.solutions.pose.PoseLandmark
        self.down_thresh = down_thresh
        self.up_thresh = up_thresh
        self.min_frames = min_frames
        self._down_frames = 0
        self._up_frames = 0

    def _knee_angle(self, landmarks, side):
        """Retorna ângulo do joelho (em graus)."""
        hip   = landmarks[getattr(self.pose, f"{side}_HIP").value]
        knee  = landmarks[getattr(self.pose, f"{side}_KNEE").value]
        ankle = landmarks[getattr(self.pose, f"{side}_ANKLE").value]

        if hip.visibility < 0.5 or knee.visibility < 0.5 or ankle.visibility < 0.5:
            return None

        a, b, c = [hip.x, hip.y, hip.z], [knee.x, knee.y, knee.z], [ankle.x, ankle.y, ankle.z]
        angle = calculate_angle_3d(a, b, c)
        return 180 - angle if angle else None  # converte para "flexão"

    def _is_valid_side_pose(self, lm):
        l_sh = lm[self.pose.LEFT_SHOULDER.value]
        r_sh = lm[self.pose.RIGHT_SHOULDER.value]

        # diferença em z (profundidade) -> indica um ombro mais perto
        z_diff = abs(l_sh.z - r_sh.z)

        # diferença em x (largura) -> fica MENOR quando está realmente de lado
        x_diff = abs(l_sh.x - r_sh.x)

        # perfil: ombros com profundidade diferente e largura mais estreita
        return z_diff > 0.10 and x_diff < 0.20

    def update(self, landmarks):
        if not self._is_valid_side_pose(landmarks):
            # Reseta pequenos estados para evitar transições falsas
            self._down_frames = 0
            self._up_frames = 0
            return None, self.counter
    
        r_flex = self._knee_angle(landmarks, "RIGHT")
        l_flex = self._knee_angle(landmarks, "LEFT")
        flexions = [f for f in (r_flex, l_flex) if f is not None]

        if not flexions:
            return None, self.counter

        # usa a média das duas pernas (mais estável que o máximo)
        current_flex = sum(flexions) / len(flexions)

        if self.stage == "up":
            if current_flex > self.down_thresh:
                self._down_frames += 1
                if self._down_frames >= self.min_frames:
                    self.stage = "down"
                    self._down_frames = 0
            else:
                self._down_frames = 0

        elif self.stage == "down":
            if current_flex < self.up_thresh:
                self._up_frames += 1
                if self._up_frames >= self.min_frames:
                    self.stage = "up"
                    self.counter += 1
                    self._up_frames = 0
            else:
                self._up_frames = 0

        return current_flex, self.counter
