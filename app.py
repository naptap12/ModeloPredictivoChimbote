import os
import warnings
import sys

# IMPORTANTE: estas variables deben setearse ANTES de importar TensorFlow
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# Configurar encoding para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D,
                                     Dense, Dropout)
from tensorflow.keras.optimizers import Adam
import gradio as gr

# ──────────────────────────────────────────────
# Parámetros del modelo (idénticos al notebook)
# ──────────────────────────────────────────────
IMG_SIZE           = 64     # Tamaño de entrada: 64x64x3
DECISION_THRESHOLD = 0.46   # Umbral optimizado obtenido en evaluación

# Rutas del modelo
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELO_DIR   = os.path.join(BASE_DIR, 'modelo')
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
    Intenta cargar el modelo guardado en distintos formatos.
    """
    # Método 1: cargar archivo .keras
    if os.path.exists(MODELO_KERAS):
        try:
            modelo = load_model(MODELO_KERAS, compile=False)
            print(f"[OK] Modelo cargado desde: {MODELO_KERAS}")
            return modelo
        except Exception as e:
            print(f"[ERROR] .keras: {e}")

    # Método 2: cargar archivo .h5 completo
    if os.path.exists(MODELO_H5):
        try:
            modelo = load_model(MODELO_H5, compile=False)
            print(f"[OK] Modelo cargado desde: {MODELO_H5}")
            return modelo
        except Exception as e:
            print(f"[ERROR] .h5: {e}")

    # Método 3: reconstruir arquitectura y cargar solo pesos
    if os.path.exists(PESOS_H5):
        try:
            modelo = reconstruir_modelo()
            modelo.load_weights(PESOS_H5)
            print(f"[OK] Arquitectura reconstruida con pesos desde: {PESOS_H5}")
            return modelo
        except Exception as e:
            print(f"[ERROR] pesos: {e}")

    return None


def predecir_imagen(img_array):
    """
    Recibe un array numpy de la imagen (entregado por Gradio),
    la preprocesa igual que en el notebook y devuelve la predicción.
    """
    global loaded_model

    if loaded_model is None:
        return "Error: modelo no cargado.", None

    try:
        # Redimensionar a 64x64 y normalizar con rescale 1./255 (igual que el notebook)
        img_resized = tf.image.resize(img_array, (IMG_SIZE, IMG_SIZE))
        img_arr     = tf.cast(img_resized, tf.float32) / 255.0
        img_arr     = tf.expand_dims(img_arr, axis=0)   # (1, 64, 64, 3)

        prob = float(loaded_model.predict(img_arr, verbose=0)[0][0])

        prob_contaminado    = prob
        prob_no_contaminado = 1 - prob

        if prob > DECISION_THRESHOLD:
            resultado = f"⚠️  CONTAMINADO  (Probabilidad: {prob_contaminado:.2%})"
        else:
            resultado = f"✅  NO CONTAMINADO  (Probabilidad: {prob_no_contaminado:.2%})"

        # Gráfico de barras de probabilidades
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        fig, ax = plt.subplots(figsize=(6, 4))
        categorias    = ['No Contaminado', 'Contaminado']
        probabilidades = [prob_no_contaminado, prob_contaminado]
        colores        = ['#4CAF50', '#f44336']

        bars = ax.bar(categorias, probabilidades, color=colores)
        ax.set_ylabel('Probabilidad')
        ax.set_title('Resultado de Predicción')
        ax.set_ylim([0, 1])

        for bar, prob_val in zip(bars, probabilidades):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{prob_val:.1%}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_plot = Image.open(buf).convert('RGB')

        return resultado, img_plot

    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}", None


# ── Cargar modelo al iniciar ──────────────────
print("Cargando modelo CNN tipo VGG...")
loaded_model = cargar_modelo()

if loaded_model is None:
    print("[ERROR] No se pudo cargar el modelo. La aplicación no funcionará correctamente.")
else:
    print("[OK] Modelo cargado exitosamente.")
    print(f"     Umbral de decisión: {DECISION_THRESHOLD}")

# ── Interfaz Gradio ───────────────────────────
with gr.Blocks(title="Detector de Contaminación", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌊 Detector de Contaminación en Imágenes

    Sube una imagen para analizar si contiene contaminación o no.

    El modelo utiliza una red neuronal convolucional (CNN) inspirada en la arquitectura
    **VGG** (Simonyan & Zisserman, 2014), entrenada desde cero para clasificación binaria.
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Sube tu imagen aquí", type="numpy")
            predict_btn = gr.Button("Analizar Imagen", variant="primary", size="lg")

        with gr.Column():
            output_text = gr.Textbox(label="Resultado", lines=2, interactive=False)
            output_plot = gr.Image(label="Probabilidades", type="pil")

    predict_btn.click(
        fn=predecir_imagen,
        inputs=input_image,
        outputs=[output_text, output_plot]
    )

    gr.Markdown("""
    ---
    **Instrucciones:**
    1. Haz clic en el área de arriba para subir una imagen o arrástrala y suéltala
    2. Presiona el botón **"Analizar Imagen"**
    3. El modelo mostrará si la imagen contiene contaminación y las probabilidades

    **Formatos soportados:** JPG, PNG, JPEG

    **Detalles técnicos:**
    - Arquitectura: CNN tipo VGG · 2 bloques convolucionales + cabeza FC
    - Entrada: 64 × 64 píxeles · Normalización: rescale 1/255
    - Umbral de decisión optimizado: 0.46
    """)

if __name__ == "__main__":
    preferred_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    try:
        demo.launch(share=False, server_name="127.0.0.1", server_port=preferred_port)
    except OSError as e:
        print(f"[WARN] No se pudo usar el puerto {preferred_port}: {e}")
        print("[INFO] Intentando con un puerto libre...")
        demo.launch(share=False, server_name="127.0.0.1", server_port=None)