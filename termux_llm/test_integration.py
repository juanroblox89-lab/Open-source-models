import os
import sys
import shutil
import traceback
from config import set_active_config, get_config
from train import train_and_save_pipeline
from quantize import quantize_model_directory
from tokenizer import BPETokenizer
from infer import generate

def run_integration_test():
    print("=" * 60)
    print("      PRUEBA DE INTEGRACIÓN OBLIGATORIA (END-TO-END)")
    print("=" * 60)
    
    # 1. Configurar la prueba para usar la configuración reducida (TEST_CONFIG)
    # Esto asegura que el entrenamiento del tokenizador y la inicialización corran en segundos.
    print("[1/5] Seleccionando configuración reducida de pruebas...")
    config = set_active_config(use_test=True)
    print(f"      vocab_size : {config['vocab_size']}")
    print(f"      n_layers   : {config['n_layers']}")
    print(f"      d_model    : {config['d_model']}")
    print("-" * 60)
    
    # Directorios temporales de prueba
    test_fp32_dir = './test_fp32_model'
    test_q4_dir = './q4_model'  # Usar el mismo q4_model que infer espera
    test_tokenizer_path = './tokenizer.json'
    
    # Limpiar cualquier directorio residual anterior
    for d in [test_fp32_dir, test_q4_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    if os.path.exists(test_tokenizer_path):
        os.remove(test_tokenizer_path)
        
    try:
        # 2. Entrenar el tokenizador y guardar tokenizer.json, inicializar pesos FP32 y guardar
        print("[2/5] Entrenando tokenizador y guardando checkpoint base (FP32)...")
        train_and_save_pipeline(config, output_dir=test_fp32_dir, tokenizer_path=test_tokenizer_path)
        print("      OK.")
        print("-" * 60)
        
        # 3. Cuantizar a Q4_0
        print("[3/5] Convirtiendo checkpoint FP32 a Q4_0 (Cuantización por bloques)...")
        quantize_model_directory(test_fp32_dir, test_q4_dir, config)
        print("      OK.")
        print("-" * 60)
        
        # 4. Cargar el tokenizador guardado y verificar alineación exacta de vocab_size
        print("[4/5] Cargando tokenizador y verificando vocab_size...")
        tok = BPETokenizer()
        tok.load(test_tokenizer_path)
        
        print(f"      Vocab del tokenizador: {tok.vocab_size}")
        print(f"      Vocab de la config: {config['vocab_size']}")
        
        if tok.vocab_size != config['vocab_size']:
            print("[-] ERROR: Desalineación crítica de vocabulario detectada.")
            sys.exit(1)
        print("      OK. El vocabulario está correctamente alineado.")
        print("-" * 60)
        
        # 5. Ejecutar la generación durante 20 tokens y confirmar decodificación libre de excepciones
        print("[5/5] Ejecutando generación de prueba (20 tokens)...")
        prompt = "El mar es"
        print(f"      Prompt de entrada: '{prompt}'")
        print("      Generando tokens: ", end='', flush=True)
        
        generated_tokens_count = 0
        for token in generate(prompt, tok, config, max_gen_len=20):
            decoded_str = tok.decode([token])
            print(decoded_str, end='', flush=True)
            generated_tokens_count += 1
            
        print(f"\n      OK. Se generaron {generated_tokens_count} tokens sin interrupciones ni excepciones.")
        print("=" * 60)
        print(" ¡PRUEBA DE INTEGRACIÓN EXITOSA! TODO EL PIPELINE FUNCIONA PERFECTAMENTE.")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "!" * 60)
        print("              FALLO EN LA PRUEBA DE INTEGRACIÓN")
        print("!" * 60)
        traceback.print_exc()
        print("!" * 60)
        sys.exit(1)
        
    finally:
        # Limpiar directorios de prueba locales para dejar el ambiente limpio
        if os.path.exists(test_fp32_dir):
            shutil.rmtree(test_fp32_dir)

if __name__ == '__main__':
    run_integration_test()
