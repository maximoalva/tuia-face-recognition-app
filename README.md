# TP1 - Sistema de Reconocimiento Facial

Plantilla base para desarrollar un sistema completo de detección, alineación, extracción de embeddings e identificación/verificación facial.

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md).

## Objetivo del backend

Implementar una API asincrónica en Python que permita:

- Registrar identidades (`/insert`)
- Ejecutar inferencia sobre una imagen (`/predict`)
- Consultar el estado del procesamiento asincrónico (`/status/{job_id}`)

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-face-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Parte 2: entrenamiento y comparación de modelos
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /insert, /predict, /status, /files
│       ├── config.py            # Configuración vía .env
│       ├── schemas.py
│       ├── services/
│       │   ├── face_service.py  # ⚠️ Funciones a implementar
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py   # Persistencia en JSON
│           └── pgvector_store.py    # Persistencia en PostgreSQL + pgvector
├── models/                      # Modelo entrenado (.pth / .onnx), no se versiona
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente local

## Requisitos para trabajar de forma local

- Python 3.12
- Docker

## Configurá tu modelo

Entrená tu modelo y guardalo dentro de la carpeta `models`. El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Recordá actualizar la configuración del `.env` correspondiente (`MODEL_NAME`, `EMBEDDING_DIM`, `FACE_SIZE`, `SIMILARITY_THRESHOLD`) para que apunte a tu modelo.

> El backend no arranca si no existe el archivo `models/<MODEL_NAME>`.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

## Opción 2 - Configurando el ambiente local

### 1. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 3. Activá el ambiente virtual

```bash
source .venv/bin/activate
```

### 4. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 5. Iniciá la base de datos

```bash
docker compose up postgres -d
```

### 6. Iniciá el frontend

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 7. Iniciá el backend

Ajustá el archivo *.env.local.example* a tus necesidades. Desde la carpeta `src`, en otra terminal:

```bash
cp ../models/<NOMBRE_DE_TU_MODELO> models/
cp ../.env.local.example .env
uvicorn app.main:app --reload --port 8000
```

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten las variables del modelo, los paths y el threshold.
3. Configuren el detector: `DETECTOR_MODEL_NAME` (opcional, archivo `.pth`/`.onnx` en `models/`), `DET_SCORE_THRESHOLD`, `DET_INPUT_SIZE`, `MIN_FACE_SIZE` y `ALIGN_MARGIN`. Si su detector viene de una librería, sobrescriban `_load_detector` en `face_service.py`.
4. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/insert`, `/predict`, `/status`) y manejo de jobs.
- Carga del modelo (`.pth` / `.onnx`) desde `models/`.
- Búsqueda por similitud configurable (`cosine` o `l2`) e identificación (`identify`).
- Manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Frontend para registrar identidades, predecir y visualizar los resultados (cajas, etiquetas y keypoints).

### A implementar por el grupo (`src/lib/services/face_service.py`)

Estas funciones hoy lanzan `NotImplementedError`:

1. `detect_faces`: detección de rostros. Devuelve las cajas `(x1, y1, x2, y2)`.
2. `align_face`: alineación con landmarks y normalización a `FACE_SIZE`. Devuelve un `AlignedFace`.
3. `extract_embedding_from_face`: extracción del embedding de dimensión `EMBEDDING_DIM`, normalizado L2.

Ver detalles y recomendaciones en [ENUNCIADO.md](ENUNCIADO.md#4-parte-3--integración-en-la-aplicación).

## Modelo y fine-tuning

El detalle completo está en `2-train.ipynb`.

**Modelo elegido:** el reconocedor `w600k_r50.onnx` del paquete `buffalo_l` de InsightFace, sin reentrenar. Es una ResNet-50 entrenada con la loss ArcFace sobre WebFace600K, con 43,6 millones de parámetros y embeddings de 512 dimensiones. El detector es `det_10g.onnx` (SCRFD-10GF) del mismo paquete. Los modelos de InsightFace son solo para investigación no comercial.

Link público de descarga (release oficial de InsightFace): https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip. Se descomprime en `models/buffalo_l/`.

**Modelo descartado:** una CNN propia (`src/lib/models/custom_cnn.py`) de 4 bloques convolucionales con BatchNorm, global average pooling, Dropout y embedding de 128 dimensiones normalizado, con 1,2 millones de parámetros. La entrenamos desde cero con las 696 personas de train como clasificador por coseno, con AdamW (lr 0,001, weight decay 0,01), ReduceLROnPlateau, early stopping y batch de 128, y augmentation solo en train. Probamos dos cabezas, softmax y ArcFace (margen 0,5), y elegimos softmax porque separó mejor a las personas nuevas de validación.

**Umbral:** `SIMILARITY_THRESHOLD=0.31` con similitud coseno. Lo calibramos en validación simulando la app (cada consulta contra la galería de personas registradas): es el umbral más bajo con el que como mucho el 1% de los desconocidos recibe un nombre. El umbral del EER de la verificación (0,19) no sirve para la app porque le daba un nombre a más de la mitad de los desconocidos.

**Métricas en test** (150 personas que ninguno de los modelos vio al entrenar):

| Métrica | CNN propia | InsightFace |
|---|---|---|
| AUC (verificación) | 0,928 | 0,99996 |
| EER | 14,9% | 0,24% |
| TAR @ FAR=1e-3 | 25,0% | 99,6% |
| Top-1 | 51,1% | 100% |
| Precision / Recall / F1 macro | 53,9% / 9,7% / 14,8% | 99,8% / 99,7% / 99,8% |
| Registrados reconocidos con su nombre | 9,7% | 99,7% |
| Desconocidos rechazados | 99,6% | 99,2% |
| Latencia en CPU (ms por rostro) | ~3 | 30 a 40 |
| Tamaño (MB) | 4,8 | 174 |

**Justificación:** elegimos InsightFace por desempeño. Con el nivel de seguridad que fijamos, la CNN propia reconoce a menos del 10% de las personas registradas, y no existe un umbral donde sus dos errores sean bajos a la vez. Es 36 veces más chica y unas 10 veces más rápida, pero la latencia de InsightFace alcanza para la app, que procesa fotos sueltas. La CNN propia rinde menos principalmente por los datos: con unas 18 fotos de 696 personas aprendió atributos generales (color de piel, pelo, género, expresión) y no los rasgos finos que distinguen a personas parecidas.

**Análisis de errores:** los falsos negativos de los dos modelos aparecen sobre todo con la cabeza girada, con rostros que eran chicos en la foto original y con cambios de apariencia (anteojos, gorras, maquillaje, pelo). Los falsos positivos de InsightFace son personas que se parecen de verdad, con similitudes cercanas al umbral. También encontramos algunas fotos de CelebA que parecen tener la etiqueta equivocada.

## Dataset

El detalle completo está en `1-pre-processing.ipynb`.

**Fuente y licencia:** usamos CelebA (Large-scale CelebFaces Attributes), de MMLAB de The Chinese University of Hong Kong, en su versión de rostros alineados (`img_align_celeba`, 178×218 px). Tiene 202.599 fotos de 10.177 celebridades. Su licencia permite usarlo solo para investigación académica no comercial y no permite redistribuirlo, así que no subimos imágenes ni datos procesados: se descarga desde la página oficial (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) y se procesa corriendo la notebook 1.

**Cantidad por persona y balance:** nos quedamos con las personas que tienen al menos 21 fotos (la mediana de CelebA) y elegimos 1.000 al azar. Después de la limpieza exigimos un mínimo de 15 fotos por persona. El dataset final tiene **25.721 imágenes de 995 personas**, con entre 15 y 30 fotos cada una (mediana 27). La persona con más fotos tiene el doble que la que menos tiene, así que no hizo falta corregir el desbalance al entrenar.

**Variaciones:** las fotos tienen variaciones reales de iluminación, pose, expresión y oclusiones (anteojos, sombreros, pelo). Las medimos con los píxeles del rostro, los puntos de la cara y los atributos anotados de CelebA. Las conservamos porque son las que el modelo tiene que tolerar.

**Reglas de filtrado y calidad:** aplicamos reglas con umbrales fijos y validados con ejemplos. Descartamos archivos corruptos, fotos sin rostro o con varios rostros ambiguos, rostros de menos de 40 px en la foto original, fotos borrosas (varianza del Laplaciano), sin color o mal expuestas, y casi-duplicados dentro de una misma persona y entre personas distintas. También descartamos las fotos que se parecen más a otra persona que a la suya según los embeddings de InsightFace (posibles etiquetas equivocadas), un par de gemelas y las personas que quedaron con menos de 15 fotos.

**Detección y alineación:** detectamos los rostros con SCRFD (`buffalo_l` de InsightFace) y los alineamos a 112×112 con los 5 puntos de la cara. Es el mismo pipeline que usa la app (`src/lib/face_pipeline.py`).

**Splits:** separamos por persona en 70% train (696 personas), 15% validación (149) y 15% test (150), así que las personas de validación y test nunca se ven al entrenar. En validación y test, unas 120 personas de cada uno se registran con 3 fotos y las otras 30 se usan como desconocidos, para evaluar el rechazo como `unknown`.

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- Los modelos (`*.pth`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen el modelo entrenado en `models/` y súbanlo a un link **público de solo lectura** para que los docentes puedan descargarlo.
