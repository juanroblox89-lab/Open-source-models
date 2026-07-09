# Transformer Edge para Termux/ARM64

Implementación de un modelo de lenguaje de ~500M parámetros ejecutable íntegramente en CPU usando únicamente Python 3.11+ y NumPy, diseñado para Android/Termux sin acceso a GPU ni toolchains de compilación pesados.

## Requisitos de Hardware
- Dispositivo Android (ej. Redmi Note 14)
- Mínimo 8 GB RAM (RAM máxima en inferencia < 1 GB garantizada por streaming)
- Almacenamiento UFS 2.2+ (para streaming eficiente)

## Instalación desde cero en Termux

1. Instalar Termux desde F-Droid (no usar versión de Google Play).
2. Actualizar paquetes base:
   ```bash
   pkg update && pkg upgrade
   ```
3. Instalar Python y dependencias de sistema:
   ```bash
   pkg install python python-pip openblas
   ```
4. Instalar librerías de Python:
   ```bash
   # NumPy en Termux generalmente se compila contra OpenBLAS automáticamente
   pip install numpy psutil
   ```
5. Comprobar aceleración BLAS:
   ```bash
   python -c "import numpy as np; np.show_config()"
   ```
   *(Asegúrate de que OpenBLAS aparezca en la configuración para garantizar velocidad de multiplicación de matrices).*

## Ejecución

1. **Generar el modelo y cuantizar (Demostración local)**
   ```bash
   cd termux_llm
   python train.py      # Genera pesos FP32 aleatorios (simula un modelo entrenado)
   python quantize.py   # Cuantiza a Q4_0 y prepara el streaming
   ```

2. **Ejecutar inferencia interactiva**
   ```bash
   python infer.py
   ```

3. **Ejecutar Benchmark**
   ```bash
   python benchmark.py
   ```

## Arquitectura (Config E)
- **d_model:** 1280
- **Capas:** 26
- **Cabezales (Q):** 20
- **Cabezales (KV):** 5 (GQA)
- **FFN Hidden:** 3520 (SwiGLU)
- **Parámetros:** ~499 Millones
- **Tamaño en disco (Q4_0):** ~250 MB
- **Streaming:** Se usa `threading` para realizar prefetch de la capa `i+1` mientras se ejecuta el `matmul` de la capa `i`, ocultando la latencia del disco.
