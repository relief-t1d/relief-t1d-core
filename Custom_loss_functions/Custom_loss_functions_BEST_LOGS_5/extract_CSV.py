import shutil
import os
from pathlib import Path

# --- CONFIGURACIÓN ---
# 1. Directorio raíz donde se ejecutará el script (el directorio actual)
ROOT_DIR = Path(os.getcwd())
# 2. Nombre de la carpeta de destino
TARGET_DIR_NAME = "CSV"
# 3. Ruta completa de la carpeta de destino
TARGET_DIR = ROOT_DIR / TARGET_DIR_NAME
# ---------------------

def copiar_csv_a_carpeta_unica():
    """
    Busca recursivamente todos los archivos CSV en el directorio raíz 
    y los copia a la carpeta de destino con sus nombres originales.
    """
    
    print(f"Buscando archivos CSV en: {ROOT_DIR}")
    
    # 1. Crear el directorio de destino si no existe
    try:
        # Crea la carpeta de destino y todos los padres intermedios si son necesarios
        TARGET_DIR.mkdir(exist_ok=True)
        print(f"Carpeta de destino asegurada: {TARGET_DIR}")
    except Exception as e:
        print(f"❌ ERROR: No se pudo crear la carpeta de destino {TARGET_DIR}. Error: {e}")
        return

    archivos_encontrados = 0
    archivos_copiados = 0
    
    # 2. Buscar recursivamente todos los archivos .csv
    # Usamos glob('**/*.csv') para búsqueda recursiva
    for filepath in ROOT_DIR.glob('**/*.csv'):
        
        # Ignorar si el archivo CSV está dentro de la propia carpeta de destino
        if TARGET_DIR_NAME in filepath.parts:
            continue
            
        archivos_encontrados += 1
        
        # 3. Construir la ruta final de destino manteniendo el nombre original
        # Solo tomamos el nombre del archivo (ej: 'resultados.csv')
        original_filename = filepath.name
        target_path = TARGET_DIR / original_filename
        
        # ** IMPORTANTE: Si dos archivos de distintas subcarpetas tienen el mismo nombre,
        # ** el último copiado sobrescribirá al anterior.
        
        try:
            # Copiar el archivo
            # shutil.copy2 copia metadatos, copy() es más simple pero copy2 es más completo.
            shutil.copy2(filepath, target_path)
            archivos_copiados += 1
            print(f"✅ Copiado: {filepath.relative_to(ROOT_DIR)} -> {original_filename}")
        except Exception as e:
            print(f"❌ ERROR al copiar {filepath.relative_to(ROOT_DIR)}: {e}")


    # 4. Resumen Final
    print("\n" + "=" * 50)
    print("📋 RESUMEN DE COPIADO")
    print("=" * 50)
    print(f"Archivos .csv encontrados: {archivos_encontrados}")
    print(f"Archivos .csv copiados: {archivos_copiados}")
    print(f"Archivos copiados en: {TARGET_DIR}")
    print("\n⚠️ ADVERTENCIA: Los archivos se copiaron con sus nombres originales. Si existen duplicados en distintas carpetas, ¡fueron sobrescritos!")
    print("=" * 50)


if __name__ == "__main__":
    copiar_csv_a_carpeta_unica()