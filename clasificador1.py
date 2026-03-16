import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import time

DATASET_FILE = "dataset_lse.csv"

CONFIDENCE_THRESHOLD = 80 
STABILITY_FRAMES = 5
COOLDOWN_FRAMES = 2
K_NEIGHBORS = 7
TEMPORAL_SMOOTHING = 4

# pesos de features
FEATURE_WEIGHTS = np.array(
    [3]*9 +      # flexiones dedos
    [2]*5 +      # distancias
    [2]*5 +      # extensiones
    [1]*3 +      # orientación
    [0.5]*3      # brazo
)


# ==========================
# FUNCIONES AUXILIARES
# ==========================

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

    return np.round(features,4)


# ==========================
# CARGAR DATASET
# ==========================

def cargar_letras():

    letras = {}

    with open(DATASET_FILE,'r',encoding='utf-8-sig') as f:

        for linea in f:

            valores = linea.strip().split(",")

            if len(valores) != 73:
                continue

            letra = valores[0].strip().lower()
            numeros = np.array(valores[1:],dtype=float)

            features = extraer_features(numeros)

            letras.setdefault(letra,[]).append(features)

    return letras


# ==========================
# CLASIFICADOR KNN
# ==========================

class Comparador:

    def __init__(self,dataset,k=5):

        self.k = k

        self.features = []
        self.labels = []

        for letra,muestras in dataset.items():

            for f in muestras:
                self.features.append(f)
                self.labels.append(letra)

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)

        # normalización
        self.mean = np.mean(self.features,axis=0)
        self.std = np.std(self.features,axis=0)+1e-6

        self.features = (self.features-self.mean)/self.std


    def comparar(self,vector):

        features = extraer_features(vector)
        features = (features-self.mean)/self.std

        # aplicar pesos
        diff = (self.features - features) * FEATURE_WEIGHTS

        dists = np.linalg.norm(diff,axis=1)

        idx = np.argsort(dists)[:self.k]

        vecinos = self.labels[idx]
        distancias = dists[idx]

        votos = {}

        for letra,dist in zip(vecinos,distancias):

            peso = 1/(dist+1e-6)

            votos[letra] = votos.get(letra,0) + peso

        total = sum(votos.values())

        resultados = {}

        for letra in votos:
            resultados[letra] = (votos[letra]/total)*100

        return resultados


# ==========================
# DETECTOR ESTABLE
# ==========================

class DetectorEstable:

    def __init__(self,confidence_threshold,stability_frames,cooldown_frames):

        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.cooldown_frames = cooldown_frames

        self.buffer = deque(maxlen=stability_frames)
        self.cooldown = 0
        self.ultima = None


    def actualizar(self,resultados):

        if not resultados:
            return None,0

        letra = max(resultados,key=resultados.get)
        conf = resultados[letra]

        if self.cooldown>0:
            self.cooldown -= 1
            return self.ultima,conf

        if conf < self.confidence_threshold:
            self.buffer.clear()
            return None,conf

        self.buffer.append(letra)

        if len(self.buffer) < self.stability_frames:
            return self.ultima,conf

        conteo = Counter(self.buffer)

        letra_estable = conteo.most_common(1)[0][0]

        if letra_estable != self.ultima:

            self.ultima = letra_estable
            self.cooldown = self.cooldown_frames

        return self.ultima,conf


# ==========================
# DIBUJAR BRAZO
# ==========================

def draw_arm(frame,results,w,h,arm_idx):

    pts=[]

    for i in arm_idx:

        lm = results.pose_landmarks.landmark[i]

        x = int(lm.x*w)
        y = int(lm.y*h)

        pts.append((x,y))

    cv2.circle(frame,pts[0],10,(0,255,255),-1)
    cv2.circle(frame,pts[1],10,(0,255,0),-1)
    cv2.circle(frame,pts[2],10,(255,0,0),-1)

    cv2.line(frame,pts[0],pts[1],(255,255,255),3)
    cv2.line(frame,pts[1],pts[2],(255,255,255),3)


# ==========================
# INICIO
# ==========================

print("Cargando dataset...")

dataset = cargar_letras()

print("Letras:",list(dataset.keys()))
print("Total muestras:",sum(len(v) for v in dataset.values()))

comparador = Comparador(dataset,k=K_NEIGHBORS)

detector = DetectorEstable(
    CONFIDENCE_THRESHOLD,
    STABILITY_FRAMES,
    COOLDOWN_FRAMES
)

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

buffer_temporal = deque(maxlen=TEMPORAL_SMOOTHING)

fps=0
t0=time.time()

with mp_holistic.Holistic(
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as holistic:

    while True:

        ret,frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame,1)

        h,w,_ = frame.shape

        t1=time.time()
        fps=0.9*fps+0.1*(1/(t1-t0+1e-6))
        t0=t1

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results = holistic.process(rgb)

        hand=None
        arm_idx=[]

        if results.right_hand_landmarks:
            hand=results.right_hand_landmarks
            arm_idx=[12,14,16]

        elif results.left_hand_landmarks:
            hand=results.left_hand_landmarks
            arm_idx=[11,13,15]

        letra=None
        confianza=0

        if hand and results.pose_landmarks:

            draw_arm(frame,results,w,h,arm_idx)

            brazo=[]

            for i in arm_idx:

                lm=results.pose_landmarks.landmark[i]
                brazo.extend([lm.x,lm.y,lm.z])

            mano=[]

            for lm in hand.landmark:
                mano.extend([lm.x,lm.y,lm.z])

            vector=np.array(brazo+mano)

            buffer_temporal.append(vector)

            vector=np.mean(buffer_temporal,axis=0)

            resultados=comparador.comparar(vector)

            letra,confianza=detector.actualizar(resultados)

            mp_draw.draw_landmarks(frame,hand,mp_holistic.HAND_CONNECTIONS)

        if letra:

            cv2.putText(frame,f"Letra: {letra}",(20,60),
                        cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

        else:

            cv2.putText(frame,"Buscando...",(20,60),
                        cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)

        cv2.putText(frame,f"Confianza: {confianza:.1f}%",(20,110),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        cv2.putText(frame,f"FPS: {fps:.1f}",(w-140,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        cv2.imshow("Detector LSE",frame)

        key=cv2.waitKey(1)&0xFF

        if key==27:
            break

cap.release()
cv2.destroyAllWindows()
