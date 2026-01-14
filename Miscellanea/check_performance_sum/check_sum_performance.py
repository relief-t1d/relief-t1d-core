import pandas as pd
import os
import glob

# --- CONFIGURACIÓN ---

# 1. Indica la carpeta donde están tus archivos CSV
#    (Usa "." para referirte a la carpeta actual)
CARPETA_DE_METRICAS = "csv_performance"

# 2. Patrón de los archivos que quieres comprobar
#    (Esto buscará cualquier archivo que empiece por "metrics_performance_by_range_"
#    y termine en ".csv")
PATRON_ARCHIVOS = "metrics_performance_by_range_*.csv"

# 3. Tolerancia permitida para la suma
#    (Ej: 0.02 permite sumas entre 99.98 y 100.02.
#    Tus ejemplos de 99.99 y 100.01 están cubiertos)
TOLERANCIA = 0.02

# -----------------------

# Columnas de porcentaje que deben sumar 100
columnas_a_sumar = ['A', 'B', 'C', 'D', 'E']

# Construir la ruta de búsqueda
ruta_busqueda = os.path.join(CARPETA_DE_METRICAS, PATRON_ARCHIVOS)
archivos_csv = glob.glob(ruta_busqueda)

if not archivos_csv:
    print(f"No se encontraron archivos con el patrón '{ruta_busqueda}'")
    print("Asegúrate de que la variable CARPETA_DE_METRICAS es correcta.")
else:
    print(f"--- Iniciando verificación de {len(archivos_csv)} archivos ---")

    archivos_con_error = 0

    for archivo in archivos_csv:
        print(f"\nVerificando archivo: {os.path.basename(archivo)}")
        try:
            df = pd.read_csv(archivo)

            # Asegurarse de que las columnas necesarias existen
            if not all(col in df.columns for col in columnas_a_sumar):
                print(f"  -> ADVERTENCIA: El archivo no contiene las columnas (A, B, C, D, E). Saltando.")
                continue

            # Calcular la suma de porcentajes en una nueva columna
            df['Suma_Total'] = df[columnas_a_sumar].sum(axis=1)

            # Encontrar filas donde la suma se salga de la tolerancia
            filas_con_error = df[abs(df['Suma_Total'] - 100) > TOLERANCIA]

            if filas_con_error.empty:
                print("  -> OK: Todas las filas suman 100 (dentro de la tolerancia).")
            else:
                archivos_con_error += 1
                print(f"  -> ¡ERROR! Se encontraron {len(filas_con_error)} filas que no suman 100:")
                # Mostramos las filas problemáticas
                print(filas_con_error[['Range', 'A', 'B', 'C', 'D', 'E', 'Suma_Total']].to_string(index=False))

        except pd.errors.EmptyDataError:
            print("  -> ADVERTENCIA: El archivo está vacío. Saltando.")
        except Exception as e:
            print(f"  -> ERROR inesperado al procesar el archivo: {e}")
            archivos_con_error += 1

    print("\n" + "="*40)
    print("--- Verificación completada ---")
    if archivos_con_error == 0:
        print("¡Todo correcto! Todos los archivos pasaron la verificación.")
    else:
        print(f"Se encontraron problemas en {archivos_con_error} archivo(s).")