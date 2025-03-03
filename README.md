# Atribus RAG - Sistema de Análisis de Datos Sociales

Sistema de Retrieval Augmented Generation (RAG) para el análisis de datos de redes sociales y la generación automática de informes.

## Descripción

Este proyecto implementa un sistema RAG para analizar datos de redes sociales y generar informes automáticos. El sistema procesa datos de diversas fuentes como Twitter, Facebook, Instagram, TikTok, YouTube, Reddit, noticias digitales, webs, foros y blogs, y utiliza técnicas de procesamiento de lenguaje natural y aprendizaje automático para extraer insights y generar informes detallados.

## Características principales

- **Procesamiento de datos multimodal**: Análisis de texto de múltiples fuentes sociales
- **Vectorstore FAISS**: Búsqueda semántica eficiente con soporte para GPU
- **Modelos LLM locales**: Integración con Ollama para procesamiento local
- **Interfaz gráfica**: Frontend intuitivo desarrollado con Streamlit
- **Generación de informes personalizados**: Análisis general, de sentimiento y de engagement
- **Procesamiento por lotes**: Manejo eficiente de grandes volúmenes de datos

## Estructura del proyecto

```
atribus_RAG/
├── backend/               # Componentes del backend
│   ├── rag_pipeline.py    # Pipeline RAG principal
│   └── report_generator.py # Generador de informes
├── data/                  # Directorio para datos
│   ├── raw/               # Datos sin procesar
│   ├── processed/         # Datos procesados
│   ├── vectorstore/       # Vectorstore para búsqueda semántica
│   └── reports/           # Informes generados
├── etl/                   # Scripts de ETL
│   ├── explore_csv.py     # Script para explorar datos CSV
│   ├── preprocess.py      # Script para preprocesar datos
│   └── create_vectorstore.py # Script para crear vectorstore
├── frontend/              # Interfaz de usuario
│   └── app.py             # Aplicación Streamlit
├── main.py                # Script principal
├── .env.example           # Ejemplo de archivo de configuración
└── README.md              # Documentación del proyecto
```

## Requisitos

- Python 3.8+
- Pandas
- LangChain y LangChain Community
- FAISS-CPU o FAISS-GPU
- Streamlit
- Ollama (servidor local para LLM)
- Loguru
- Python-dotenv
- Torch
- Transformers (HuggingFace)

## Instalación

1. Clona este repositorio:
   ```
   git clone https://github.com/tu-usuario/atribus_RAG.git
   cd atribus_RAG
   ```

2. Crea un entorno virtual:
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

4. Copia el archivo de configuración:
   ```
   cp .env.example .env
   ```

5. Edita el archivo `.env` con tus configuraciones.

6. Asegúrate de tener Ollama instalado y los modelos descargados:
   ```
   # Instala Ollama desde https://ollama.com/
   # Descarga los modelos necesarios
   ollama pull granite3.2
   ollama pull granite-embedding:278m
   ```

## Uso

### Preparación de datos

1. Coloca tu archivo CSV de datos en el directorio `data/raw/`.

2. Explora los datos:
   ```
   python main.py explore
   ```

3. Preprocesa los datos:
   ```
   python main.py preprocess
   ```

4. Crea el vectorstore:
   ```
   python main.py vectorstore
   ```

### Ejecución de la aplicación

Para ejecutar la aplicación Streamlit:
```
python main.py run
```

### Pipeline completo

Para ejecutar todo el pipeline (preprocesamiento, vectorstore y aplicación):
```
python main.py pipeline
```

## Tipos de informes

El sistema puede generar tres tipos de informes:

1. **General**: Análisis general de los datos con estadísticas y tendencias principales.
2. **Sentiment**: Enfocado en análisis de sentimiento, identificando opiniones positivas, negativas y neutrales.
3. **Engagement**: Enfocado en métricas de engagement, interacciones y alcance.

## Estructura de datos

El sistema espera un CSV con los siguientes campos (entre otros):

- `TEXTO`: Contenido textual del post/comentario
- `FECHA`: Fecha de publicación (se mantiene con hora incluida para análisis temporales detallados)
- `FUENTE`: Plataforma de origen (Twitter, Facebook, etc.)
- `URL`: URL del post/comentario
- `ENGAGEMENT`: Métricas de engagement (soporta formato numérico con coma como separador decimal)
- `CATEGORÍA`: Categoría o tema del contenido

Durante el preprocesamiento, se generan campos adicionales como:

- `ID_UNICO`: Identificador único para cada registro
- `FECHA_ISO`: Fecha normalizada en formato ISO
- `TEXTO_ENRIQUECIDO`: Campo de texto enriquecido para RAG

## Personalización

Puedes personalizar el comportamiento del sistema editando los parámetros en el archivo `.env`:

- `LLM_MODEL`: Modelo de LLM a utilizar (por defecto: granite3.2)
- `EMBEDDING_MODEL`: Modelo de embeddings (por defecto: granite-embedding:278m)
- `CHUNK_SIZE`: Tamaño de los chunks para el vectorstore
- `CHUNK_OVERLAP`: Solapamiento entre chunks
- `TOP_K`: Número de documentos relevantes a recuperar
- `DATA_PATH`: Ruta al directorio de datos
- `VECTORSTORE_PATH`: Ruta al directorio del vectorstore

## Solución de problemas

### Problemas comunes

1. **Error al cargar el vectorstore**: Verifica que los archivos index.faiss e index.pkl existan en el directorio data/vectorstore.
2. **Errores de memoria con datasets grandes**: Utiliza el procesamiento por lotes ajustando BATCH_SIZE en el archivo .env.
3. **Problemas con Ollama**: Asegúrate de que el servidor Ollama esté en ejecución y los modelos estén correctamente descargados.

### Logs

El sistema utiliza Loguru para registrar información detallada. Revisa los logs para diagnosticar problemas:

```
tail -f logs/atribus_rag.log
```

## Licencia

[Incluir información de licencia]

## Contacto

[Incluir información de contacto]
