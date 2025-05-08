import subprocess
import os

#1)Configura aquí tus rutas (ajusta al servidor)
work_dir       = r"C:\Users\Jonathan Lema\Desktop\archivos"
script_name    = "predict_similitud.py"
artefacto      = "artefacto_similitud.joblib"
asuntos_excel  = "asuntos_simulados.xlsx"
output_xlsx    = "resultados_produccion.xlsx"
threshold      = "0.25"

#2)Cambia al directorio donde está todo
os.chdir(work_dir)

#3)Construye el comando de llamada
cmd = [
    "python", script_name,
    artefacto,
    asuntos_excel,
    "--threshold", threshold,
    "--output", output_xlsx
]

#4)Ejecútalo y captura salida
proc = subprocess.run(cmd, capture_output=True, text=True)
if proc.returncode != 0:
    #Si falla, lanzamos excepción para que Rocketbot lo marque como error
    raise RuntimeError(f"Error al ejecutar {script_name}:\n{proc.stderr}")

#5)Si llega aquí, todo fue bien
print(f"[+] {script_name} ejecutado correctamente. "
      f"Salida: {output_xlsx}")