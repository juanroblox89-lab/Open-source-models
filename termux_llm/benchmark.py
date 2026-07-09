import os
import time
import resource
import subprocess
import psutil

def get_memory_usage():
    # Returns memory usage in MB
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
    return total / 1024 / 1024

if __name__ == '__main__':
    print("Iniciando Benchmark Termux Edge...")
    
    if os.path.exists('./q4_model'):
        size_mb = get_dir_size('./q4_model')
        print(f"Tamaño en disco (Q4_0): {size_mb:.2f} MB")
    
    print(f"RAM base: {get_memory_usage():.2f} MB")
    
    # Run inference as a subprocess to measure peak clean
    print("Ejecutando prueba de latencia (5 tokens)...")
    cmd = "python infer.py"
    
    t0 = time.time()
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    t1 = time.time()
    
    # This is a basic wrapper. The actual detailed stats are printed by infer.py
    out_str = stdout.decode()
    for line in out_str.split('\n'):
        if '[Stats]' in line:
            print(line)
            
    print(f"RAM aproximada tras ejecución: {get_memory_usage():.2f} MB")
    print("Nota: Para medir RAM pico real en Termux, observar con 'top' o 'htop' durante la ejecución.")
