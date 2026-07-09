# Transformer Edge para Termux/ARM64 (Redmi Note 14 / ARM64 CPUs)

Implementación optimizada de un modelo autoregresivo transformer decoder-only de **~499 millones de parámetros (Configuración E)** ejecutable al 100% en CPU móvil utilizando únicamente Python 3.11+ y NumPy. Específicamente diseñado para ejecutarse dentro de **Termux sobre Android 14** sin root, sin dependencias de GPU/NPU, y sin compiladores nativos complejos.

## Requisitos de Hardware
- **Dispositivo**: Procesador ARM64 de gama media o superior (ej. MediaTek Dimensity 6100+ en Redmi Note 14).
- **RAM Física**: Mínimo 8 GB (el pico de RAM en inferencia se mantiene < 1.0 GB gracias a la arquitectura de streaming por capas y cuantización Q4_0).
- **Almacenamiento**: UFS 2.2+ (para lecturas de disco veloces durante el streaming).

---

## Características de la Arquitectura (Configuración E)
- **d_model**: 1280 (residual stream width)
- **Capas (Layers)**: 26 (prioriza profundidad para razonamiento complejo)
- **Atención (GQA)**: Grouped Query Attention con ratio 4:1 (20 cabezales de Query, 5 de Key/Value; `head_dim` = 64) para reducir drásticamente el KV-cache.
- **FFN Hidden (SwiGLU)**: 3520 (activación SwiGLU con factor de escala ~2.75x para rendimiento superior).
- **Normalización**: RMSNorm dual por capa (sin bias, máxima eficiencia matemática).
- **Embeddings Posicionales**: RoPE (Rotary Positional Embeddings) con base theta=10000.
- **Ventana de Atención**: Sliding Window de 512 tokens (limita la complejidad cuadrática en secuencias largas).
- **Pesos Compartidos (Tied Embeddings)**: Los embeddings de entrada y de proyección de salida comparten memoria, ahorrando ~41M de parámetros.

---

## Instalación Paso a Paso en Termux

1. **Instalar Termux**: Se recomienda descargar el APK desde **F-Droid** (la versión de Google Play Store está desactualizada y no tiene repositorios de paquetes funcionales).
2. **Actualizar el sistema de paquetes**:
   ```bash
   pkg update && pkg upgrade -y
   ```
3. **Instalar Python y OpenBLAS**:
   ```bash
   pkg install python python-pip openblas ndk-sysroot clang -y
   ```
4. **Instalar NumPy y dependencias adicionales**:
   ```bash
   # NumPy compilará o enlazará automáticamente con OpenBLAS en ARM64
   pip install numpy psutil
   ```
5. **Verificar la aceleración BLAS**:
   ```bash
   python -c "import numpy as np; np.show_config()"
   ```
   *(Asegúrate de que OpenBLAS aparezca en las secciones de la biblioteca matemática para garantizar velocidad de multiplicación de matrices).*

---

## Pipeline de Ejecución

Todos los hiperparámetros se configuran desde un único archivo centralizado `config.py` que actúa como la **única fuente de verdad** del proyecto.

### 1. Ejecutar la Prueba de Integración Obligatoria (End-to-End)
Antes de proceder con el modelo grande, ejecuta la prueba de validación del pipeline. Utiliza una configuración reducida para entrenar el tokenizador, inicializar el modelo, cuantizar a Q4_0 y generar texto en segundos:
```bash
python test_integration.py
```
*Este script fallará ruidosamente si existe alguna desalineación en el tamaño del vocabulario o en las dimensiones del modelo.*

### 2. Generar el Checkpoint FP32 y Tokenizador de Producción
Genera los pesos maestros base aleatorios en precisión FP32 y entrena el tokenizador BPE completo de 32,000 palabras:
```bash
python train.py
```
*Los pesos se guardarán fragmentados por capa en `./fp32_model` para facilitar la cuantización.*

### 3. Cuantizar el Modelo Completo a Q4_0
Reduce los pesos de 32 bits a bloques empaquetados de 4 bits con signo, logrando un ahorro de tamaño del 87.5% en disco (~250 MB en total):
```bash
python quantize.py
```
*Los archivos finales se optimizan y guardan bajo la estructura `./q4_model`.*

### 4. Lanzar la Inferencia Interactiva
Inicia la generación de texto con prefetch asíncrono y doble buffer (lectura paralela del disco en hilos secundarios para ocultar la latencia de almacenamiento UFS):
```bash
python infer.py
```

### 5. Medir Rendimiento con el Benchmark
Genera estadísticas precisas de tiempo hasta el primer token, velocidad continua de generación (tokens/segundo), tamaño exacto en disco y consumo máximo de RAM pico:
```bash
python benchmark.py
```

---

## Análisis de Viabilidad Técnica

> **Estrategia de Streaming de Capas (Streaming on-the-fly):**
> Dado que NumPy y OpenBLAS no admiten operaciones matriciales en INT8 o Q4 nativos en ARM64 de forma acelerada, tener los pesos cuantizados directamente en memoria no aceleraría el cálculo por sí mismo. Por ende, la arquitectura implementada carga las capas cuantizadas de disco (`UFS 2.2`), las descuantiza asíncronamente en hilos independientes (`threading`) a un búfer FP32 temporal, ejecuta la multiplicación `@` altamente optimizada por OpenBLAS en CPU, y libera el buffer antes de pasar a la siguiente capa.
> 
> Esta técnica desacopla el tamaño del modelo de la memoria RAM activa, permitiendo ejecutar un modelo de 500M de parámetros utilizando **menos de 1.0 GB de RAM pico**.
