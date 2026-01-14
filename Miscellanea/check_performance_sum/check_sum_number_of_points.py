import pandas as pd
import os
import glob
import numpy as np

# --- CONFIGURACIÓN ---

# 1. Indica la carpeta donde están tus archivos CSV
CARPETA_DE_DATOS = "csv_number_points"

# 2. Patrón de los archivos que quieres comprobar
PATRON_ARCHIVOS = "number_of points_by_zones_*.csv"

# 3. Lista EXACTA de las filas de los sub-rangos que deben sumarse
RANGOS_A_SUMAR = ['TBR_2', 'TBR_1', 'TIR', 'TAR_1', 'TAR_2']

# 4. Columnas que se deben sumar y comprobar
COLUMNAS_A_COMPROBAR = ['A', 'B', 'C', 'D', 'E']

# -----------------------

ruta_busqueda = os.path.join(CARPETA_DE_DATOS, PATRON_ARCHIVOS)
archivos_csv = glob.glob(ruta_busqueda)

if not archivos_csv:
    print(f"No se encontraron archivos con el patrón '{ruta_busqueda}'")
    print("Asegúrate de que la variable CARPETA_DE_DATOS es correcta.")
else:
    print(f"--- Iniciando verificación de {len(archivos_csv)} archivos ---")
    print(f"Se sumarán las filas: {RANGOS_A_SUMAR}\n")

    archivos_con_error = 0

    for archivo in archivos_csv:
        print(f"Verificando archivo: {os.path.basename(archivo)}")
        try:
            # 1. Leer el CSV, auto-detectando el separador
            df = pd.read_csv(archivo, sep=None, engine='python')

            # 2. Limpiar columna 'Range'
            if 'Range' in df.columns:
                df['Range'] = df['Range'].str.strip()
            else:
                print(f"  -> ERROR: No se encontró la columna 'Range'.")
                print(f"     Columnas encontradas: {df.columns.tolist()}")
                archivos_con_error += 1
                continue

            # --- INICIO DE LA CORRECCIÓN DE TIPO ---
            # 3. Forzar la conversión de las columnas A-E a números
            columnas_existen = True
            for col in COLUMNAS_A_COMPROBAR:
                if col not in df.columns:
                    print(f"  -> ADVERTENCIA: Falta la columna {col}. Saltando archivo.")
                    columnas_existen = False
                    break

                # Si la columna NO es numérica (es 'object'/'string')...
                if not pd.api.types.is_numeric_dtype(df[col]):
                    # ...reemplazar comas (de miles) y espacios, luego convertir
                    df[col] = (
                        df[col].astype(str)
                        .str.replace(',', '', regex=False)
                        .str.strip()
                    )

                # Convertir a numérico (int), los valores no válidos (NaN) se ponen a 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.int64)

            if not columnas_existen:
                continue  # Salta al siguiente archivo
            # --- FIN DE LA CORRECCIÓN DE TIPO ---

            # --- Lógica de la suma ---

            fila_entire = df[df['Range'] == 'ENTIRE']

            if fila_entire.empty:
                print(f"  -> ADVERTENCIA: No se encontró la fila 'ENTIRE'. Saltando.")
                continue

            filas_rangos = df[df['Range'].isin(RANGOS_A_SUMAR)]

            if len(filas_rangos) != len(RANGOS_A_SUMAR):
                print(
                    f"  -> ADVERTENCIA: No se encontraron las {len(RANGOS_A_SUMAR)} filas de rango esperadas. Saltando.")
                continue

            # Esta suma ahora SÍ es de enteros
            suma_vertical_calculada = filas_rangos[COLUMNAS_A_COMPROBAR].sum()
            valores_entire = fila_entire.iloc[0][COLUMNAS_A_COMPROBAR]

            # 5. Comparar
            if (suma_vertical_calculada == valores_entire).all():
                print("  -> OK: La suma de los 5 rangos coincide perfectamente con la fila 'ENTIRE'.")
            else:
                archivos_con_error += 1
                print(f"  -> ¡ERROR! La suma de los 5 rangos NO coincide con la fila 'ENTIRE'.")

                print("\n     --- Detalles del Desajuste (Columna por Columna) ---")
                df_comparacion = pd.DataFrame({
                    'Valor_ENTIRE': valores_entire,
                    'Suma_Calculada_Rangos': suma_vertical_calculada
                })
                df_comparacion['Diferencia'] = df_comparacion['Valor_ENTIRE'] - df_comparacion['Suma_Calculada_Rangos']
                print(df_comparacion[df_comparacion['Diferencia'] != 0])
                print("     -------------------------------------------------\n")

        except Exception as e:
            print(f"  -> ERROR inesperado al procesar el archivo: {e}")
            archivos_con_error += 1

    print("\n" + "=" * 40)
    print("--- Verificación completada ---")
    if archivos_con_error == 0:
        print("¡Todo correcto! Todos los archivos pasaron la verificación.")
    else:
        print(f"Se encontraron problemas en {archivos_con_error} archivo(s).")