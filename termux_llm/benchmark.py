import os
import sys
import time
import resource
import subprocess
import psutil
from config import get_config

def get_memory_usage():
    # Retorna el uso de memoria actual del proceso en MB
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_dir_size(path):
    if not os.path.exists(path):
        return 0.0
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
    return total / 1024 / 1024

if __name__ == '__main__':
    config = get_config()
    print("=" * 55)
    print("          BENCHMARK SYSTEM: TRANSFORMER EDGE")
    print("=" * 55)
    
    # Mostrar variables de entorno de hilos activas
    print("[*] CONFIGURACIÓN DE MULTIHILO (Álgebra Lineal):")
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        print(f"    - {var}: {os.environ.get(var, 'No definida (Usando valor de sistema)')}")
    print("-" * 55)
    
    # 1. Tamaño en disco
    q4_size = get_dir_size('./q4_model')
    fp32_size = get_dir_size('./fp32_model')
    print(f"[*] Tamaño en disco (FP32 Maestro) : {fp32_size:.2f} MB")
    print(f"[*] Tamaño en disco (Q4_0 Cuantizado): {q4_size:.2f} MB")
    
    # 2. RAM Inicial
    print(f"[*] Uso de RAM inicial (Intérprete Python): {get_memory_usage():.2f} MB")
    
    # 3. Medición de Inferencia y Rendimiento de Generación
    print("\nEjecutando inferencia en subproceso...")
    cmd = f"{sys.executable} infer.py"
    
    t0 = time.time()
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    t1 = time.time()
    
    # Analizar y reportar métricas capturadas del subproceso
    out_str = stdout.decode('utf-8', errors='replace')
    err_str = stderr.decode('utf-8', errors='replace')
    
    print("\n--- SALIDA DE LA GENERACIÓN ---")
    lines = out_str.split('\n')
    for line in lines:
        if "Transformer Edge: " in line or "Usuario: " in line:
            print(line)
        elif "[Stats]" in line:
            print(line)
            
    if err_str.strip():
        print("\n--- ERRORES ENCONTRADOS ---")
        print(err_str)
        
    # 4. Medición de RAM Pico del subproceso (RUSAGE_CHILDREN)
    # resource.getrusage en Linux retorna la memoria pico (maxrss) en kilobytes (KB)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_rss_mb = usage.ru_maxrss / 1024.0
    
    print("\n" + "=" * 55)
    print("                RESULTADOS FINALES")
    print("=" * 55)
    print(f"[*] Consumo de RAM Pico de Inferencia: {peak_rss_mb:.2f} MB")
    print(f"[*] Tiempo total de ejecución (Subproceso): {t1 - t0:.2f} s")
    print("=" * 55)
