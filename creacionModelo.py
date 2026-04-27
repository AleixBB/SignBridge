import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# FUNCIONES DE EXTRACCIÓN DE FEATURES (IDÉNTICAS A LAS DE CLASIFICADOR.PY)

def angulo(a, b, c):
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosang, -1, 1))

def angulo_vectores(v1, v2):
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.arccos(np.clip(cosang, -1, 1))

def extraer_features(vector):
    """
    Extrae features geométricas invariantes a escala y posición
    Input: vector de 72 dimensiones (9 de brazo + 63 de mano)
    Output: vector de features (~52 dimensiones)
    """
    vector = np.array(vector, dtype=float)
    
    # Separar brazo y mano
    brazo_raw = vector[:9].reshape(3, 3)  # 3 puntos (hombro, codo, muñeca) x 3 coordenadas
    mano_raw = vector[9:].reshape(21, 3)  # 21 puntos de la mano x 3 coordenadas
    
    # --- NORMALIZACIÓN: centrar en la muñeca y escalar ---
    muneca = mano_raw[0]  # El primer punto de la mano es la muñeca
    
    # Restar la muñeca a todos los puntos de la mano
    mano_normalizada = mano_raw - muneca
    
    # Escalar basado en la distancia del dedo corazón (punto 12 es el MCP del corazón)
    distancia_referencia = np.linalg.norm(mano_normalizada[12])
    if distancia_referencia > 0.01:
        mano_normalizada = mano_normalizada / distancia_referencia
    
    # Reconstruir vector normalizado (usamos mano normalizada pero brazo sin normalizar)
    # El brazo mantiene sus valores relativos
    vector_normalizado = np.concatenate([brazo_raw.flatten(), mano_normalizada.flatten()])
    
    # Extraer nuevamente después de normalizar
    brazo = vector_normalizado[:9].reshape(3, 3)
    mano = vector_normalizado[9:].reshape(21, 3)
    
    # Centrar en la muñeca nuevamente (por si acaso)
    puntos = mano.copy()
    puntos -= puntos[0]  # Muñeca en origen
    
    # Escalar para hacer las features invariantes al tamaño
    escala = np.linalg.norm(puntos[9])  # Distancia al MCP del corazón
    if escala != 0: 
        puntos /= escala
    
    # ==========================================
    # 1. Ángulos de flexión de dedos (9 features)
    # ==========================================
    # Dedos: índice, medio, anular, meñique
    # Cada dedo tiene 2 ángulos (PIP y DIP)
    dedos = [(5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]
    flexiones = []
    
    for mcp, pip, dip, tip in dedos:
        # Ángulo en la articulación PIP (entre MCP y PIP)
        flexiones.append(angulo(puntos[mcp], puntos[pip], puntos[dip]))
        # Ángulo en la articulación DIP (entre PIP y DIP)
        flexiones.append(angulo(puntos[pip], puntos[dip], puntos[tip]))
    
    # Ángulo del pulgar (entre MCP del pulgar, articulación y punta)
    flexiones.append(angulo(puntos[2], puntos[3], puntos[4]))
    
    # ==========================================
    # 2. Distancias entre dedos (5 features)
    # ==========================================
    distancias = [
        np.linalg.norm(puntos[8] - puntos[12]),    # Distancia índice - medio
        np.linalg.norm(puntos[12] - puntos[16]),   # Distancia medio - anular
        np.linalg.norm(puntos[16] - puntos[20]),   # Distancia anular - meñique
        np.linalg.norm(puntos[4] - puntos[8]),     # Distancia pulgar - índice
        np.linalg.norm(puntos[4] - puntos[12])     # Distancia pulgar - medio
    ]
    
    # ==========================================
    # 3. Extensiones de dedos (5 features)
    # ==========================================
    extensiones = [
        np.linalg.norm(puntos[8]),   # Extensión del índice
        np.linalg.norm(puntos[12]),  # Extensión del medio
        np.linalg.norm(puntos[16]),  # Extensión del anular
        np.linalg.norm(puntos[20]),  # Extensión del meñique
        np.linalg.norm(puntos[4])    # Extensión del pulgar
    ]
    
    # ==========================================
    # 4. Ángulos entre dedos (5 features)
    # ==========================================
    angulos_dedos = [
        angulo_vectores(puntos[8], puntos[12]),   # Ángulo índice-medio
        angulo_vectores(puntos[12], puntos[16]),  # Ángulo medio-anular
        angulo_vectores(puntos[16], puntos[20]),  # Ángulo anular-meñique
        angulo_vectores(puntos[8], puntos[16]),   # Ángulo índice-anular
        angulo_vectores(puntos[8], puntos[20])    # Ángulo índice-meñique
    ]
    
    # ==========================================
    # 5. Orientación de la palma (3 features)
    # ==========================================
    # Vector desde la base del índice hasta la base del meñique
    v1 = puntos[5] - puntos[0]   # Base índice - muñeca
    v2 = puntos[17] - puntos[0]  # Base meñique - muñeca
    normal = np.cross(v1, v2)    # Vector normal a la palma
    
    norm_norm = np.linalg.norm(normal)
    if norm_norm > 1e-6:
        orientacion = normal / norm_norm
    else:
        orientacion = np.zeros(3)
    
    # ==========================================
    # 6. Features del brazo (3 features)
    # ==========================================
    sh, el, wr = brazo  # Hombro, codo, muñeca
    brazo_f = [
        np.linalg.norm(sh - el),      # Longitud antebrazo
        np.linalg.norm(el - wr),      # Longitud brazo
        angulo(sh, el, wr)            # Ángulo del codo
    ]
    
    # ==========================================
    # COMBINAR TODAS LAS FEATURES
    # ==========================================
    features = np.concatenate([flexiones, distancias, extensiones, angulos_dedos, orientacion, brazo_f])
    
    # Verificar y limpiar valores problemáticos
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        print(f"⚠️ Advertencia: Se encontraron valores NaN/Inf en features. Reemplazando con 0.")
        features = np.nan_to_num(features, 0)
    
    return features

# ==========================================
# CARGA DEL DATASET
# ==========================================

def cargar_dataset(csv_path="dataset_lse.csv"):
    """
    Carga el dataset desde CSV y extrae features para todas las muestras
    """
    print("="*60)
    print("CARGANDO DATASET")
    print("="*60)
    
    features_list = []
    labels_list = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            lineas = f.readlines()
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo {csv_path}")
        print("   Asegúrate de que el archivo existe en el directorio actual")
        return None, None
    
    print(f"📄 Archivo encontrado: {csv_path}")
    print(f"📊 Total de líneas: {len(lineas)}")
    
    muestras_validas = 0
    muestras_invalidas = 0
    
    for idx, linea in enumerate(lineas):
        linea = linea.strip()
        if not linea:
            continue
            
        valores = linea.split(",")
        
        # Verificar que tenga al menos 73 columnas (letra + 72 coordenadas)
        if len(valores) < 73:
            muestras_invalidas += 1
            continue
        
        letra = valores[0].strip().lower()
        
        # Verificar que la letra no esté vacía
        if not letra:
            muestras_invalidas += 1
            continue
        
        try:
            # Tomar las primeras 72 coordenadas después de la letra
            coordenadas = np.array(valores[1:73], dtype=float)
            
            # Extraer features
            features = extraer_features(coordenadas)
            
            features_list.append(features)
            labels_list.append(letra)
            muestras_validas += 1
            
        except (ValueError, IndexError) as e:
            muestras_invalidas += 1
            if muestras_invalidas < 5:  # Mostrar solo primeros 5 errores
                print(f"   ⚠️ Error en línea {idx+1}: {e}")
            continue
        
        # Progreso
        if (idx + 1) % 100 == 0:
            print(f"   Procesadas {idx + 1}/{len(lineas)} muestras...")
    
    print(f"\n✅ Carga completada:")
    print(f"   - Muestras válidas: {muestras_validas}")
    print(f"   - Muestras inválidas: {muestras_invalidas}")
    
    if muestras_validas == 0:
        print("❌ ERROR: No se encontraron muestras válidas en el dataset")
        return None, None
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"   - Dimensiones features: {X.shape}")
    print(f"   - Clases únicas: {np.unique(y)}")
    
    # Mostrar distribución de clases
    print("\n📊 Distribución de clases:")
    clases, counts = np.unique(y, return_counts=True)
    for clase, count in zip(clases, counts):
        print(f"   {clase}: {count} muestras")
    
    return X, y

# ==========================================
# ENTRENAMIENTO DEL MODELO
# ==========================================

def entrenar_modelo(X, y):
    """
    Entrena el modelo Random Forest
    """
    print("\n" + "="*60)
    print("ENTRENANDO MODELO")
    print("="*60)
    
    # Verificar datos
    if X is None or y is None:
        return None, None
    
    if len(X) == 0:
        print("❌ ERROR: No hay datos para entrenar")
        return None, None
    
    # Escalar características
    print("📏 Escalando características...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"   - Media de features después de escalar: {X_scaled.mean(axis=0)[:5]}...")
    print(f"   - Std de features después de escalar: {X_scaled.std(axis=0)[:5]}...")
    
    # Configurar y entrenar Random Forest
    print("\n🌲 Entrenando Random Forest...")
    modelo = RandomForestClassifier(
        n_estimators=100,        # Número de árboles
        max_depth=15,            # Profundidad máxima de cada árbol
        min_samples_split=5,     # Mínimo de muestras para dividir un nodo
        min_samples_leaf=2,      # Mínimo de muestras en una hoja
        random_state=42,         # Semilla para reproducibilidad
        n_jobs=-1,               # Usar todos los núcleos disponibles
        verbose=1                # Mostrar progreso
    )
    
    modelo.fit(X_scaled, y)
    
    # Evaluación
    train_score = modelo.score(X_scaled, y)
    print(f"\n📈 Precisión en entrenamiento: {train_score:.4f} ({train_score*100:.2f}%)")
    
    # Importancia de features
    importancias = modelo.feature_importances_
    print(f"\n🔝 Top 10 características más importantes:")
    top_idx = np.argsort(importancias)[-10:][::-1]
    for i, idx in enumerate(top_idx):
        print(f"   {i+1}. Feature {idx}: {importancias[idx]:.4f}")
    
    return modelo, scaler

# ==========================================
# GUARDAR MODELO
# ==========================================

def guardar_modelo(modelo, scaler, clases, path="modelo_lse_rf.pkl"):
    """
    Guarda el modelo entrenado en un archivo pickle
    """
    print("\n" + "="*60)
    print("GUARDANDO MODELO")
    print("="*60)
    
    data = {
        'modelo': modelo,
        'scaler': scaler,
        'tipo': 'random_forest',
        'classes': list(clases),
        'feature_dim': modelo.n_features_in_ if hasattr(modelo, 'n_features_in_') else None
    }
    
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Modelo guardado exitosamente en: {path}")
        print(f"   - Tipo: {data['tipo']}")
        print(f"   - Clases: {len(data['classes'])} clases")
        if data['feature_dim']:
            print(f"   - Dimensión de features: {data['feature_dim']}")
        
        # Verificar que se pueda cargar
        print("\n🔍 Verificando integridad del modelo guardado...")
        with open(path, 'rb') as f:
            test_load = pickle.load(f)
        print("   ✅ Modelo se puede cargar correctamente")
        
    except Exception as e:
        print(f"❌ Error al guardar el modelo: {e}")
        return False
    
    return True

# ==========================================
# PRUEBA RÁPIDA DEL MODELO
# ==========================================

def probar_modelo(modelo, scaler):
    """
    Prueba el modelo con un vector sintético
    """
    print("\n" + "="*60)
    print("PRUEBA RÁPIDA DEL MODELO")
    print("="*60)
    
    # Crear vector de prueba (simulando una mano)
    vector_prueba = np.zeros(72)
    
    # Simular brazo
    vector_prueba[0:3] = [0.5, 0.5, 0.5]   # Hombro
    vector_prueba[3:6] = [0.4, 0.4, 0.4]   # Codo
    vector_prueba[6:9] = [0.3, 0.3, 0.3]   # Muñeca
    
    # Simular mano (valores pequeños y variados)
    for i in range(9, 72, 3):
        vector_prueba[i] = 0.2 * (i % 5) / 5.0
        vector_prueba[i+1] = 0.2 * ((i+1) % 5) / 5.0
        vector_prueba[i+2] = 0.1 * ((i+2) % 3) / 3.0
    
    print("📝 Probando con vector sintético...")
    
    try:
        # Extraer features
        features = extraer_features(vector_prueba)
        print(f"   Features extraídas: {len(features)} dimensiones")
        
        # Escalar
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predecir
        if hasattr(modelo, 'predict_proba'):
            probabilidades = modelo.predict_proba(features_scaled)[0]
            mejor_idx = np.argmax(probabilidades)
            mejor_letra = modelo.classes_[mejor_idx]
            mejor_prob = probabilidades[mejor_idx] * 100
            
            print(f"\n✅ Prueba exitosa!")
            print(f"   Letra predicha: {mejor_letra.upper()}")
            print(f"   Confianza: {mejor_prob:.2f}%")
            
            # Mostrar top 3
            top3_idx = np.argsort(probabilidades)[-3:][::-1]
            print(f"   Top 3 predicciones:")
            for idx in top3_idx:
                print(f"     - {modelo.classes_[idx].upper()}: {probabilidades[idx]*100:.2f}%")
        else:
            letra = modelo.predict(features_scaled)[0]
            print(f"\n✅ Prueba exitosa!")
            print(f"   Letra predicha: {letra.upper()}")
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================

def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("\n" + "="*60)
    print("REENTRENAMIENTO URGENTE DEL MODELO LSE")
    print("="*60)
    print("\nEste script reentrenará el modelo Random Forest")
    print("con la versión mejorada de extracción de features.\n")
    
    # 1. Cargar dataset
    X, y = cargar_dataset("dataset_lse.csv")
    
    if X is None or y is None:
        print("\n❌ No se pudo cargar el dataset. Verifica que el archivo existe.")
        print("   El archivo debe llamarse 'dataset_lse.csv' y estar en el mismo directorio.")
        return
    
    # 2. Entrenar modelo
    modelo, scaler = entrenar_modelo(X, y)
    
    if modelo is None:
        print("\n❌ No se pudo entrenar el modelo.")
        return
    
    # 3. Guardar modelo
    clases = np.unique(y)
    exito = guardar_modelo(modelo, scaler, clases, "modelo_lse_rf.pkl")
    
    if not exito:
        print("\n❌ No se pudo guardar el modelo.")
        return
    
    # 4. Probar modelo
    probar_modelo(modelo, scaler)
    
    # 5. Resumen final
    print("\n" + "="*60)
    print("="*60)
    print("✅ Modelo entrenado y guardado correctamente")
    print("✅ Archivo generado: modelo_lse_rf.pkl")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()