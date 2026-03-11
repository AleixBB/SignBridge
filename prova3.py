import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# ==========================
# FEATURES Y PESOS
# ==========================

NOMBRES_FEATURES = [
    "flex_index_1","flex_index_2",
    "flex_middle_1","flex_middle_2",
    "flex_ring_1","flex_ring_2",
    "flex_pinky_1","flex_pinky_2",
    "flex_thumb",
    "dist_index_middle",
    "dist_middle_ring",
    "dist_ring_pinky",
    "dist_thumb_index",
    "dist_thumb_middle",
    "ext_index",
    "ext_middle",
    "ext_ring",
    "ext_pinky",
    "ext_thumb",
    "palm_orientation_x",
    "palm_orientation_y",
    "palm_orientation_z",
    "arm_length",
    "forearm_length",
    "elbow_angle"
]

PESOS_FEATURES = np.array([
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.2,0.8,0.8,1.2
])

# ==========================
# FUNCIONES AUXILIARES
# ==========================

def espejo_x(x, flip):
    return 1-x if flip else x


def angulo(a,b,c):
    ba = a-b
    bc = c-b
    cosang = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    cosang = np.clip(cosang,-1,1)
    return np.arccos(cosang)


def extraer_features(vector):

    vector = np.array(vector)

    brazo = vector[:9].reshape(3,3)
    mano = vector[9:].reshape(21,3)

    shoulder, elbow, wrist_arm = brazo

    puntos = mano.copy()
    wrist = puntos[0]

    puntos -= wrist

    escala = np.linalg.norm(puntos[9])
    if escala != 0:
        puntos /= escala

    dedos = [(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)]

    flexiones = []

    for mcp,pip,dip,tip in dedos:
        flexiones.append(angulo(puntos[mcp],puntos[pip],puntos[dip]))
        flexiones.append(angulo(puntos[pip],puntos[dip],puntos[tip]))

    flexiones.append(angulo(puntos[2],puntos[3],puntos[4]))

    distancias = [
        np.linalg.norm(puntos[8]-puntos[12]),
        np.linalg.norm(puntos[12]-puntos[16]),
        np.linalg.norm(puntos[16]-puntos[20]),
        np.linalg.norm(puntos[4]-puntos[8]),
        np.linalg.norm(puntos[4]-puntos[12])
    ]

    extensiones = [np.linalg.norm(puntos[i]) for i in [8,12,16,20,4]]

    v1 = puntos[5]-puntos[0]
    v2 = puntos[17]-puntos[0]

    normal = np.cross(v1,v2)
    orientacion = normal/(np.linalg.norm(normal)+1e-6)

    arm_length = np.linalg.norm(shoulder-elbow)
    forearm_length = np.linalg.norm(elbow-wrist_arm)
    elbow_angle = angulo(shoulder,elbow,wrist_arm)

    brazo_features = [arm_length,forearm_length,elbow_angle]

    features = np.concatenate([
        flexiones,
        distancias,
        extensiones,
        orientacion,
        brazo_features
    ])

    return features


def cargar_letras():

    letras = {}

    with open('/Users/aleixbenet/Desktop/PRACTICA LIS/dataset_lse.csv','r',encoding='utf-8-sig') as f:

        for linea in f:

            linea = linea.strip()

            if not linea:
                continue

            letra = linea.split(",")[0]

            numeros = [float(x) for x in linea.split(",")[1:]]

            if len(numeros) == 72:

                features = extraer_features(numeros)

                if letra not in letras:
                    letras[letra] = []

                letras[letra].append(features)

    return letras


class Comparador:

    def __init__(self,dataset):
        self.refs = {letra: np.mean(muestras,axis=0) for letra,muestras in dataset.items()}

    def comparar(self,vector):

        features = extraer_features(vector)

        resultados = {}
        errores = {}
        confianza_features = {}

        for letra, ref in self.refs.items():

            dif = np.abs(features-ref)*PESOS_FEATURES
            dist = np.linalg.norm(dif)

            resultados[letra] = max(0,100-dist*40)

            confianza_features[letra] = [max(0,100-d*100) for d in dif]

            peor_idx = np.argmax(dif)

            errores[letra] = (NOMBRES_FEATURES[peor_idx], dif[peor_idx])

        return resultados, errores, confianza_features


# ==========================
# INICIO
# ==========================

print("Cargando dataset...")

dataset = cargar_letras()

if not dataset:
    print("❌ No se pudieron cargar letras")
    exit()

print("Letras disponibles:", list(dataset.keys()))

comparador = Comparador(dataset)

# ==========================
# MEDIAPIPE
# ==========================

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

buffer = deque(maxlen=9)

with mp_holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
) as holistic:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame,1)

        h,w,_ = frame.shape

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results = holistic.process(rgb)

        hand = None
        arm_idx = []
        flip_x = False

        if results.right_hand_landmarks:
            hand = results.right_hand_landmarks
            arm_idx = [12,14,16]

        elif results.left_hand_landmarks:
            hand = results.left_hand_landmarks
            arm_idx = [11,13,15]
            flip_x = True

        if hand and results.pose_landmarks:

            brazo = []
            pts = []

            for i in arm_idx:

                lm = results.pose_landmarks.landmark[i]
                x = lm.x
                #x = espejo_x(lm.x, flip_x)
                y = lm.y
                z = lm.z

                brazo.extend([x,y,z])

                pts.append((int(x*w), int(y*h)))

            cv2.circle(frame, pts[0], 12, (0,255,255), -1)
            cv2.circle(frame, pts[1], 12, (0,255,0), -1)
            cv2.circle(frame, pts[2], 12, (255,0,0), -1)

            cv2.line(frame, pts[0], pts[1], (255,255,255), 4)
            cv2.line(frame, pts[1], pts[2], (255,255,255), 4)

            mano = []

            for lm in hand.landmark:

                x = espejo_x(lm.x, flip_x)

                mano.extend([x,lm.y,lm.z])

            vector = np.array(brazo+mano)

            resultados, errores, confianza_features = comparador.comparar(vector)

            mejor = max(resultados, key=resultados.get)

            buffer.append(mejor)

            letra = max(set(buffer), key=buffer.count)

            cv2.putText(frame,f"Letra: {letra}",(20,60),
                        cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

            conf = resultados[mejor]

            cv2.putText(frame,f"Confianza: {conf:.1f}%",(20,100),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            caracteristica,_ = errores[mejor]

            cv2.putText(frame,f"Error principal: {caracteristica}",(20,130),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

            y=170

            for i,nombre in enumerate(NOMBRES_FEATURES):

                conf_val = confianza_features[mejor][i]

                cv2.putText(frame,f"{nombre}: {conf_val:.0f}%",
                            (20,y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255,255,255),
                            1)

                y+=18

            mp_draw.draw_landmarks(frame, hand, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow("Detector de Letras", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()