import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter, defaultdict
import time

DATASET_FILE = "dataset_lse.csv"

CONFIDENCE_THRESHOLD = 75
STABILITY_FRAMES = 4
COOLDOWN_FRAMES = 0
K_NEIGHBORS = 11
TEMPORAL_SMOOTHING = 3

# 🔥 NUEVOS UMBRALES
UNKNOWN_THRESHOLD = 90
MARGIN_THRESHOLD = 15

AVAILABLE_LETTERS = set(['a','b','f','t','c','d','e','i','k','l','m','n','o','p','q','r','s','u'])

FEATURE_WEIGHTS = np.array(
    [2.5]*9 + #flexiones
    [5]*5 + #distancias
    [2.5]*5 + #extensiones
    [2]*5 + #angulos
    [1.5]*3 + #distancias relativas
    [0.5]*3 #angulos relativos
)

# ==========================
# DICCIONARIO + AUTOCORRECTOR
# ==========================
def cargar_diccionario_archivo(path, letras_validas):
    palabras_validas = set()
    with open(path, 'r', encoding='utf-8') as f:
        for linea in f:
            palabra = linea.strip().lower()
            if not palabra.isalpha():
                continue
            if all(l in letras_validas for l in palabra):
                palabras_validas.add(palabra)
    return sorted(palabras_validas)

def distancia_levenshtein(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j

    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            coste = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j]+1,
                dp[i][j-1]+1,
                dp[i-1][j-1]+coste
            )
    return dp[-1][-1]

def indexar_diccionario(diccionario):
    index = defaultdict(list)
    for palabra in diccionario:
        index[len(palabra)].append(palabra)
    return index

def autocorregir(palabra):
    if not palabra:
        return palabra

    mejor = palabra
    mejor_dist = float("inf")

    for l in range(len(palabra)-2, len(palabra)+3):
        for w in DICT_INDEX.get(l, []):
            dist = distancia_levenshtein(palabra, w)
            if dist < mejor_dist:
                mejor_dist = dist
                mejor = w

    return mejor if mejor_dist <= 2 else palabra

# ==========================
# FEATURES
# ==========================
def angulo(a,b,c):
    ba = a-b
    bc = c-b
    cosang = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    cosang = np.clip(cosang,-1,1)
    return np.arccos(cosang)

def angulo_vectores(v1, v2):
    cosang = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
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

    v_index = puntos[8]
    v_middle = puntos[12]
    v_ring = puntos[16]
    v_pinky = puntos[20]

    angulos_entre_dedos = [
        angulo_vectores(v_index, v_middle),
        angulo_vectores(v_middle, v_ring),
        angulo_vectores(v_ring, v_pinky),
        angulo_vectores(v_index, v_ring),
        angulo_vectores(v_index, v_pinky)
    ]

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
        angulos_entre_dedos,
        orientacion,
        brazo_features
    ])

    return np.round(features,4)

# ==========================
# DATASET
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
# KNN
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

        self.mean = np.mean(self.features,axis=0)
        self.std = np.std(self.features,axis=0)+1e-6

        self.features = (self.features-self.mean)/self.std

    def comparar(self,vector):
        features = extraer_features(vector)
        features = (features-self.mean)/self.std

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

        # 🔥 FILTRO DE INCERTIDUMBRE
        ordenadas = sorted(resultados.items(), key=lambda x: x[1], reverse=True)
        if len(ordenadas) == 0:
            return {}
        top1, conf1 = ordenadas[0]
        if conf1 < UNKNOWN_THRESHOLD:
            return {}
        if len(ordenadas) > 1:
            top2, conf2 = ordenadas[1]
            if (conf1 - conf2) < MARGIN_THRESHOLD:
                return {}
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
            self.buffer.clear()
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
# DRAW ARM
# ==========================
def draw_arm(frame,results,w,h,arm_idx):
    pts=[]
    for i in arm_idx:
        lm = results.pose_landmarks.landmark[i]
        x = int(lm.x*w)
        y = int(lm.y*h)
        pts.append((x,y))

    cv2.circle(frame,pts[0],8,(0,255,255),-1)
    cv2.circle(frame,pts[1],8,(0,255,0),-1)
    cv2.circle(frame,pts[2],8,(255,0,0),-1)

    cv2.line(frame,pts[0],pts[1],(255,255,255),3)
    cv2.line(frame,pts[1],pts[2],(255,255,255),3)

# ==========================
# INICIO
# ==========================
print("Cargando dataset...")
dataset = cargar_letras()

print("Cargando diccionario...")
DICTIONARY = cargar_diccionario_archivo("dict.txt", AVAILABLE_LETTERS)
DICT_INDEX = indexar_diccionario(DICTIONARY)

comparador = Comparador(dataset,k=K_NEIGHBORS)
detector = DetectorEstable(CONFIDENCE_THRESHOLD,STABILITY_FRAMES,COOLDOWN_FRAMES)

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
buffer_temporal = deque(maxlen=TEMPORAL_SMOOTHING)

current_word = []
palabra_cruda = ""
palabra_final = ""
no_hand_counter = 0
ultima_letra = None
NO_HAND_FRAMES_THRESHOLD = 15

#Variables tiempo
TIEMPO_LETRA = 0.5
tiempo_inicio_letra = None
letra_en_proceso = None

# ==========================
# LOOP
# ==========================
with mp_holistic.Holistic() as holistic:
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame,1)
        h,w,_ = frame.shape

        results = holistic.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

        hand = results.right_hand_landmarks or results.left_hand_landmarks
        arm_idx = [12,14,16] if results.right_hand_landmarks else [11,13,15]

        letra=None
        confianza=0

        tiempo_actual = time.time()

        if hand and results.pose_landmarks:

            no_hand_counter = 0
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

            # 🔥 LOGICA TIEMPO 0.8s
            if letra and confianza > UNKNOWN_THRESHOLD:
                if letra != letra_en_proceso:
                    letra_en_proceso = letra
                    tiempo_inicio_letra = tiempo_actual
                else:
                    if tiempo_inicio_letra and (tiempo_actual - tiempo_inicio_letra >= TIEMPO_LETRA):
                        if letra != ultima_letra:
                            current_word.append(letra)
                            ultima_letra = letra
                        tiempo_inicio_letra = None
                        letra_en_proceso = None
            else:
                letra_en_proceso = None
                tiempo_inicio_letra = None

            mp_draw.draw_landmarks(frame,hand,mp_holistic.HAND_CONNECTIONS)

        else:
            no_hand_counter += 1
            letra_en_proceso = None
            tiempo_inicio_letra = None

            if no_hand_counter > NO_HAND_FRAMES_THRESHOLD and current_word:
                palabra_cruda = "".join(current_word)
                palabra_final = autocorregir(palabra_cruda)
                current_word = []
                buffer_temporal.clear()
                detector.buffer.clear()
                ultima_letra = None

        # ==========================
        # DIBUJAR INFO
        # ==========================
        cv2.putText(frame,f"Letra: {letra if letra else '-'}",(20,60),
                    cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        cv2.putText(frame,f"Confianza: {confianza:.1f}%",(20,110),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        cv2.putText(frame,f"Palabra: {''.join(current_word)}",(20,160),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
        cv2.putText(frame,f"Raw: {palabra_cruda}",(20,210),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(frame,f"Auto: {palabra_final}",(20,250),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)

        cv2.imshow("SignBridge",frame)
        if cv2.waitKey(1)==27:
            break

cap.release()
cv2.destroyAllWindows()
