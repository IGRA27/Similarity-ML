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

Para la licencia solicita informacion a SoftConsulting S.A.

# Acerca del modelo:
No es un modelo entrenado en el sentido clásico (como una red neuronal finetuneada). Contiene:

| Clave         | Descripción |
|---------------|-------------|
| **`model_name`** | Nombre del modelo Sentence-BERT utilizado (`all-MiniLM-L6-v2`). |
| **`syn_texts`**  | 2 000 cadenas del tipo:<br>`"Anexo Aclaratorio EQUIPOS ELECTRÓNICOS | Anexo aclaratorio EE | …"` |
| **`syn_emb`**    | Matriz `2000 × 384` (*float32*) con los *embeddings* de cada cadena. |
| **`meta_df`**    | Columnas originales *Tipo de Movimiento* y *Ramo*, como referencia. |

En memoria ocupa aproximadamente 3 MB (`2000 × 384 × 4 bytes`).

Durante la inferencia, el script:
1. Genera un embedding SBERT para **cada asunto**.
2. Calcula la similitud de coseno contra `syn_emb` (2 000 dimensiones).
3. Ordena y devuelve los **Top-k** resultados.

---

### 2. ¿Qué tan “poderoso” es?

| Factor                | Valor con `all-MiniLM-L6-v2` |
|------------------------|-----------------------------|
| **Calidad semántica**  | Muy decente para textos cortos (≈ 85 % de precisión de modelos más grandes). |
| **Velocidad en CPU**   | ≈ 10 000 textos en 15 segundos (laptop moderna). |
| **Memoria en producción** | ≈ 550 MB para el modelo + 3 MB para `syn_emb`. |
| **Idiomas**            | Entrenado principalmente en **inglés**; funciona en español, pero con pérdida de matices. |
| **Tamaño del catálogo** | Escalable hasta decenas de miles de filas (O(n) en tiempo y memoria). |

> Para *matching* de frases cortas y vocabulario limitado en seguros, ofrece un **punto dulce** entre velocidad y precisión.

---

### 3. Limitaciones

1. **No está finetuneado** al dominio de seguros en Ecuador, por lo que pierde contexto legal y local.  
2. **Modelo en inglés**: detecta similitudes en español gracias al aprendizaje multilingüe, pero no tan bien como uno entrenado en español.  
3. **No aprende de nuevas entradas**: si cambian las expresiones (“endoso”, “siniestro parcial”, etc.), hay que **reentrenar** el artefacto.  
4. **No da probabilidades calibradas**: un score de 0.75 **no** significa 75 % de certeza, solo que es más alto que, por ejemplo, 0.60.

---

### 4. Cómo medir “qué tan bueno” es

1. Construye un **set de validación** (ej. 300 asuntos reales con *Movimiento* y *Ramo* etiquetados).  
2. Ejecuta `predict_top3.py` y calcula:

- **Top-1 accuracy** = % de asuntos cuyo `Match_1` es correcto.  
- **Top-3 accuracy** = % donde la categoría correcta está entre los 3 primeros.

3. Sin finetuning, un objetivo razonable es:  
   - **Top-1** ≈ 70 %  
   - **Top-3** > 85 %

---

### 5. Cómo hacerlo “más poderoso”

| Mejora                                           | Resultado esperado                        | Esfuerzo     |
|--------------------------------------------------|-------------------------------------------|--------------|
| **Cambiar a `all-mpnet-base-v2` (768 dims)**     | +5–8 puntos de precisión, ~2× RAM/tiempo  | 1 línea      |
| **Modelo multilingüe `paraphrase-multilingual-mpnet-base-v2`** | Mejor manejo del español                  | 1 línea      |
| **Finetuning supervisado** (con tripletas)       | +10–15 puntos en Top-1                    | Alto         |
| **Normalización de texto** (lowercase, sin tildes, stopwords) | +2–4 puntos                          | Bajo         |
| **Uso de vector DB / FAISS**                     | Búsquedas sub-segundo en catálogos grandes| Medio        |
| **Umbral + fallback “Otros”**                    | Menos falsos positivos                    | Bajo         |

---

### 6. Resumen rápido para TI / DevOps

1. **Artefacto liviano (~3 MB)** → fácil de versionar y transportar.  
2. **CPU-friendly** → corre en VM con 2 vCPU / 4 GB RAM, sin GPU.  
3. **Determinista** → mismo catálogo ⇒ mismo output. Solo se reentrena al cambiar el catálogo.  
4. **Escalable** → si el catálogo crece, simplemente regeneras `artefacto_similitud.joblib`.

En resumen: es un **motor de búsqueda semántico especializado** para tu catálogo de movimientos. Lo suficientemente bueno para un MVP en producción y lo bastante liviano para hardware modesto, con un camino claro de mejoras a medida que crece el proyecto.
