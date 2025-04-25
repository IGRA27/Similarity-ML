# Proyecto de Similitud de Asuntos con SBERT

Este repositorio contiene la implementación de un pipeline para calcular la similitud semántica entre asuntos de correos electrónicos utilizando **Sentence-BERT (SBERT)**. Está pensado para integrarse con un bot RPA (Rocketbot) que envía diariamente dos archivos Excel y recibe a cambio un reporte con los porcentajes de similitud.

---

## Tabla de Contenidos

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Uso](#uso)
  - [Preprocesamiento de Datos](#preprocesamiento-de-datos)
  - [Entrenamiento del Modelo (opcional)](#entrenamiento-del-modelo-opcional)
  - [Cálculo de Similitud](#cálculo-de-similitud)
- [Integración con Rocketbot](#integración-con-rocketbot)
- [Salida](#salida)
- [Evaluación](#evaluación)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

---

## Características

- **SBERT** para embeddings semánticos de alta calidad.
- Pipeline completo de ingestión, preprocesamiento, cálculo de similitud y exportación.
- Soporte para archivos Excel de entrada y salida con `openpyxl`.
- Reporte de resultados con porcentaje de similitud.
- Integración sencilla con bots RPA (Rocketbot).

---

## Requisitos

- Python 3.8 o superior
- Paquetes:
  - `sentence-transformers`
  - `pandas`
  - `openpyxl`
  - `tqdm`

---

## Instalación

```
# Clonar el repositorio
git clone https://github.com/IGRA27/Similarity-ML.git
cd .

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

```

# Uso
Uso
Preprocesamiento de Datos (opcional)
Entrenamiento del Modelo 
Si deseas afinar el modelo SBERT o entrenar un clasificador adicional:

python src/train_model.py \
  --data data/train_clean.xlsx \
  --model_output models/custom_sbert

# Cálculo de Similitud
Ejecuta el script principal indicando las rutas de los archivos de entrada y la ruta de salida:

python predict_similitud.py \
  --input_a ruta/primer_archivo.xlsx \
  --input_b ruta/segundo_archivo.xlsx \
  --output ruta/resultados.xlsx
input_a y input_b deben contener columnas Asunto, ID, Correo.

El archivo de salida incluirá además la columna Similitud con el porcentaje.

## Integración con RPA
1. Configura el bot para pasar dos variables con las rutas locales de los Excel.

2. Invoca el script Python anterior.

3. El bot recogerá el archivo resultados.xlsx y lo enviará según tu flujo.

### Salida
El archivo Excel de salida (resultados.xlsx) tendrá estas columnas:
Asunto	ID	Correo	Similitud (%)
Por favor ayudar con trámite...	22864	user100@empresa.com	57.46

#### Evaluación
Puedes evaluar la calidad de la similitud midiendo:

Correlación con anotaciones humanas (Pearson/Spearman).

Histograma de distribuciones de similitud.

Threshold óptimo para clasificaciones.

##### Contribuciones
Haz un fork del repositorio.

Crea una rama (git checkout -b feature/nueva-funcionalidad).

Haz commit de tus cambios (git commit -m 'Agrega nueva funcionalidad').

Sube la rama (git push origin feature/nueva-funcionalidad).

Abre un Pull Request.

Licencia
