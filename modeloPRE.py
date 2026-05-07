import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import warnings
import sys

# Configurar encoding para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Suprimir warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

print("[OK] Warnings de TensorFlow y Python suprimidos.")

# ──────────────────────────────────────────────
# Parámetros del modelo (idénticos al notebook)
# ──────────────────────────────────────────────
IMG_SIZE         = 64          # Tamaño de entrada: 64x64x3
DECISION_THRESHOLD = 0.46      # Umbral optimizado obtenido en evaluación

# Rutas del modelo
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELO_DIR  = os.path.join(BASE_DIR, 'modelo')
MODELO_KERAS = os.path.join(MODELO_DIR, 'modelo_cnn_contaminacion.keras')
MODELO_H5    = os.path.join(MODELO_DIR, 'modelo_cnn_contaminacion.h5')
PESOS_H5     = os.path.join(MODELO_DIR, 'pesos_modelo.weights.h5')


def reconstruir_modelo():
    """
    Reconstruye la arquitectura CNN inspirada en VGG (Simonyan & Zisserman, 2014)
    exactamente como fue definida en el notebook de entrenamiento.

    Arquitectura:
        Bloque 1 — Extracción de características (primera pasada):
            Conv2D(16, 3x3, ReLU) → MaxPooling2D(2x2) → Dropout(0.4)

        Bloque 2 — Profundización de características (segunda pasada):
            Conv2D(32, 3x3, ReLU) → MaxPooling2D(2x2) → Dropout(0.5)

        Cabeza clasificadora (FC layers):
            GlobalAveragePooling2D → Dense(32, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)

    Entrada: 64 x 64 x 3 (RGB, normalizado con rescale 1./255)
    Salida : probabilidad ∈ [0, 1] — umbral de decisión: 0.46
    """
    print("\n--- Reconstruyendo arquitectura CNN tipo VGG ---")

    model = Sequential([
        # ── Bloque 1: Extracción de características ──
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        # ── Bloque 2: Profundización de características ──
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.5),

        # ── Cabeza clasificadora ──
        GlobalAveragePooling2D(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    return model


def cargar_modelo():
    """
    Intenta cargar el modelo guardado. Si no lo encuentra en ningún formato,
    reconstruye la arquitectura y carga los pesos por separado.
    """
    # Método 1: cargar archivo .keras
    if os.path.exists(MODELO_KERAS):
        print(f"\n[1] Intentando cargar modelo .keras...")
        try:
            modelo = load_model(MODELO_KERAS, compile=False)
            print(f"[OK] Modelo cargado desde: {MODELO_KERAS}")
            return modelo
        except Exception as e:
            print(f"[ERROR] {e}")

    # Método 2: cargar archivo .h5 completo
    if os.path.exists(MODELO_H5):
        print(f"\n[2] Intentando cargar modelo .h5...")
        try:
            modelo = load_model(MODELO_H5, compile=False)
            print(f"[OK] Modelo cargado desde: {MODELO_H5}")
            return modelo
        except Exception as e:
            print(f"[ERROR] {e}")

    # Método 3: reconstruir arquitectura y cargar solo los pesos
    if os.path.exists(PESOS_H5):
        print(f"\n[3] Reconstruyendo arquitectura y cargando pesos...")
        try:
            modelo = reconstruir_modelo()
            modelo.load_weights(PESOS_H5)
            print(f"[OK] Arquitectura reconstruida con pesos desde: {PESOS_H5}")
            return modelo
        except Exception as e:
            print(f"[ERROR] {e}")

    print("\n[ERROR] No se encontró ningún archivo de modelo en la carpeta 'modelo/'.")
    return None


def predecir_contaminacion(img_path, modelo=None, threshold=DECISION_THRESHOLD):
    """
    Predice si una imagen es Contaminada o No Contaminada.

    Args:
        img_path  : Ruta a la imagen (.jpg / .png)
        modelo    : Instancia del modelo cargado (usa el global si es None)
        threshold : Umbral de decisión (por defecto 0.46, el optimizado en evaluación)

    Returns:
        dict con clase, probabilidades y umbral usado
    """
    if modelo is None:
        raise ValueError("No se proporcionó un modelo.")

    # Cargar y preprocesar imagen — igual que en el notebook (rescale 1./255)
    img     = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_arr = image.img_to_array(img)
    img_arr = img_arr / 255.0                          # rescale 1./255
    img_arr = np.expand_dims(img_arr, axis=0)          # (1, 64, 64, 3)

    prob = float(modelo.predict(img_arr, verbose=0)[0][0])

    return {
        'clase'                      : 'Contaminado' if prob > threshold else 'No Contaminado',
        'contaminado'                : prob > threshold,
        'probabilidad_contaminado'   : round(prob, 4),
        'probabilidad_no_contaminado': round(1 - prob, 4),
        'threshold_usado'            : threshold
    }


# ── Main ──────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("CARGADOR DE MODELO PREDICTIVO DE CONTAMINACIÓN")
    print("Arquitectura: CNN tipo VGG (Simonyan & Zisserman, 2014)")
    print("=" * 60)

    loaded_model = cargar_modelo()

    if loaded_model:
        loaded_model.summary()
        print("\n[OK] Modelo listo para realizar predicciones.")
        print(f"     Umbral de decisión: {DECISION_THRESHOLD}")

        # Ejemplo de uso — ajusta esta ruta a una imagen real
        test_img_path = 'ruta/a/tu/imagen.jpg'

        if os.path.exists(test_img_path):
            print(f"\n--- Predicción de prueba: {test_img_path} ---")
            resultado = predecir_contaminacion(test_img_path, loaded_model)
            print("\nResultado:")
            for k, v in resultado.items():
                print(f"  {k}: {v}")
        else:
            print(f"\n[INFO] Ajusta 'test_img_path' a una imagen válida para probar.")
    else:
        print("\n[ERROR] No se pudo cargar el modelo con ningún método.")
        print("  Verifica que la carpeta 'modelo/' contenga alguno de estos archivos:")
        print("    - modelo_cnn_contaminacion.keras")
        print("    - modelo_cnn_contaminacion.h5")
        print("    - pesos_modelo.weights.h5")