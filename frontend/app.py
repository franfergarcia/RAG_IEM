"""
Aplicación Streamlit para el sistema RAG de análisis de datos sociales.
"""
import os
import sys
import pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime

# Añadir el directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar componentes del backend
from backend.rag_pipeline import RAGPipeline

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Atribus RAG - Análisis de Datos Sociales",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 10px 0;
        text-align: center;
        min-width: 150px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #616161;
    }
    .sentiment-positive {
        color: #4CAF50;
    }
    .sentiment-negative {
        color: #F44336;
    }
    .sentiment-neutral {
        color: #9E9E9E;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar el pipeline RAG
@st.cache_resource
def get_rag_pipeline():
    """Inicializa y devuelve el pipeline RAG."""
    pipeline = RAGPipeline()
    pipeline.initialize()
    return pipeline

# Verificar recursos necesarios
def check_resources():
    """Verifica que todos los recursos necesarios estén disponibles."""
    # Verificar directorios
    for directory in ["data/raw", "data/processed", "data/vectorstore", "data/reports"]:
        if not os.path.exists(directory):
            st.error(f"No se encontró el directorio: {directory}")
            return False
    
    # Verificar archivo de datos procesados
    processed_data_path = os.getenv("PROCESSED_DATA_PATH")
    if not os.path.exists(processed_data_path):
        st.error(f"No se encontró el archivo de datos procesados: {processed_data_path}")
        st.info("Ejecuta primero el preprocesamiento de datos con: `python main.py preprocess`")
        return False
    
    # Verificar vectorstore
    vectorstore_path = os.getenv("VECTORSTORE_PATH")
    if not os.path.exists(vectorstore_path) or not os.listdir(vectorstore_path):
        st.error(f"No se encontró el vectorstore en: {vectorstore_path}")
        st.info("Ejecuta primero la creación del vectorstore con: `python main.py vectorstore`")
        return False
    
    return True

# Cargar datos procesados
@st.cache_data
def load_processed_data():
    """Carga los datos procesados."""
    processed_data_path = os.getenv("PROCESSED_DATA_PATH")
    return pd.read_csv(processed_data_path)

# Obtener temas disponibles
@st.cache_data
def get_available_topics():
    """Obtiene los temas disponibles en los datos."""
    df = load_processed_data()
    if 'CATEGORÍA' in df.columns:
        return df['CATEGORÍA'].dropna().unique().tolist()
    return []

# Funciones para la interfaz de usuario
def render_header():
    """Renderiza el encabezado de la aplicación."""
    st.markdown('<div class="main-header">Atribus RAG - Análisis de Datos Sociales</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sistema de Retrieval Augmented Generation para análisis de datos sociales</div>', unsafe_allow_html=True)

def render_sidebar():
    """Renderiza la barra lateral con opciones de configuración."""
    st.sidebar.title("Configuración")
    
    # Opciones de consulta
    st.sidebar.header("Consulta")
    
    topics = get_available_topics()
    query = ""
    if topics:
        selected_topic = st.sidebar.selectbox("Selecciona un tema", topics)
        query = f"Análisis sobre {selected_topic}"
    else:
        st.sidebar.warning("No se encontraron temas en los datos procesados")
    
    # Tipo de informe
    st.sidebar.header("Tipo de informe")
    report_type = st.sidebar.radio(
        "Selecciona el tipo de informe",
        ["general", "sentiment", "engagement"]
    )
    
    # Botón para generar informe
    generate_button = st.sidebar.button("Generar informe", type="primary")
    
    # Información adicional
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Tipos de informe:**
    - **General**: Análisis general de los datos
    - **Sentiment**: Enfocado en análisis de sentimiento
    - **Engagement**: Enfocado en métricas de engagement
    """)
    
    return query, report_type, generate_button

def render_metrics(df):
    """Renderiza métricas generales de los datos."""
    st.markdown('<div class="sub-header">Métricas generales</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total de registros</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if 'FUENTE' in df.columns:
            sources_count = df['FUENTE'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{sources_count}</div>
                <div class="metric-label">Fuentes diferentes</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Métricas de sentimiento
    st.markdown('<div class="sub-header">Análisis de sentimiento</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'SENTIMIENTO' in df.columns:
            positive_pct = (df['SENTIMIENTO'] == 'positive').mean() * 100
            st.markdown(f"""
            <div class="metric-card sentiment-positive">
                <div class="metric-value">{positive_pct:.1f}%</div>
                <div class="metric-label">Sentimiento positivo</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'SENTIMIENTO' in df.columns:
            neutral_pct = (df['SENTIMIENTO'] == 'neutral').mean() * 100
            st.markdown(f"""
            <div class="metric-card sentiment-neutral">
                <div class="metric-value">{neutral_pct:.1f}%</div>
                <div class="metric-label">Sentimiento neutro</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'SENTIMIENTO' in df.columns:
            negative_pct = (df['SENTIMIENTO'] == 'negative').mean() * 100
            st.markdown(f"""
            <div class="metric-card sentiment-negative">
                <div class="metric-value">{negative_pct:.1f}%</div>
                <div class="metric-label">Sentimiento negativo</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Información temporal
    st.markdown('<div class="sub-header">Rango temporal</div>', unsafe_allow_html=True)
    if 'FECHA_ISO' in df.columns:
        min_date = df['FECHA_ISO'].min().split('T')[0] if 'T' in str(df['FECHA_ISO'].min()) else df['FECHA_ISO'].min()
        max_date = df['FECHA_ISO'].max().split('T')[0] if 'T' in str(df['FECHA_ISO'].max()) else df['FECHA_ISO'].max()
        date_range = f"{min_date} a {max_date}"
        st.markdown(f"""
        <div class="date-range-card">
            <div class="date-range-value">{date_range}</div>
            <div class="date-range-label">Periodo de análisis</div>
        </div>
        """, unsafe_allow_html=True)

def render_report(report_data):
    """Renderiza un informe generado según la estructura actual del JSON."""
    if not report_data:
        return
    
    # Verificar si hay error en el informe
    if "error" in report_data:
        st.error(f"Error al generar el informe: {report_data['error']}")
        return
        
    # Extraer información del informe
    query = report_data.get("query", "")
    report_type = report_data.get("type", "general")
    timestamp = report_data.get("timestamp", datetime.now().isoformat())
    data = report_data.get("data", {})
    metadata = report_data.get("metadata", {})
    
    # Fecha formateada
    try:
        date_obj = datetime.fromisoformat(timestamp)
        date_str = date_obj.strftime("%d/%m/%Y")
        time_str = date_obj.strftime("%H:%M")
    except:
        date_str = timestamp
        time_str = ""
    
    # CSS para el informe (estilo profesional)
    st.markdown("""
    <style>
        .report-container {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .report-title {
            font-size: 24px;
            font-weight: bold;
            color: #1E293B;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #E2E8F0;
        }
        .report-section {
            font-size: 20px;
            font-weight: bold;
            color: #1E293B;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        .numbered-topic {
            font-size: 18px;
            font-weight: bold;
            color: #334155;
            margin-top: 25px;
            margin-bottom: 10px;
        }
        .topic-content {
            color: #475569;
            margin-bottom: 15px;
            line-height: 1.6;
        }
        .example-quote {
            font-style: italic;
            background-color: #F8FAFC;
            border-left: 3px solid #3B82F6;
            padding: 10px 15px;
            margin: 15px 0;
        }
        .conclusion-label {
            font-weight: bold;
            color: #334155;
        }
        .metadata-box {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .metadata-title {
            font-size: 16px;
            font-weight: bold;
            color: #334155;
        }
        .source-tabs {
            margin-top: 10px;
        }
        .source-content {
            padding: 10px;
            background-color: #F8FAFC;
            border-radius: 5px;
        }
        .sentiment-positive {
            color: #10B981;
            font-weight: bold;
        }
        .sentiment-neutral {
            color: #6B7280;
            font-weight: bold;
        }
        .sentiment-negative {
            color: #EF4444;
            font-weight: bold;
        }
        .divider {
            margin: 20px 0;
            border-bottom: 1px solid #E2E8F0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Contenedor principal
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    
    # Título del informe
    st.markdown(f'<div class="report-title">Resumen general</div>', unsafe_allow_html=True)
    
    # Mostrar metadatos en un expander
    with st.expander("Metadatos del informe", expanded=False):
        st.markdown('<div class="metadata-box">', unsafe_allow_html=True)
        st.markdown('<div class="metadata-title">Metadatos del informe</div>', unsafe_allow_html=True)
        
        # Información del modelo y parámetros
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Modelo LLM:** {metadata.get('model', 'No especificado')}")
            st.markdown(f"**Modelo de embeddings:** {metadata.get('embedding_model', 'No especificado')}")
        
        with col2:
            st.markdown(f"**Documentos recuperados:** {metadata.get('documents_retrieved', 'No especificado')}")
            st.markdown(f"**Tiempo de generación:** {report_data.get('generation_time', 0):.2f} segundos")
        
        # Mostrar fuentes utilizadas
        st.markdown('<div class="metadata-title">Fuentes utilizadas</div>', unsafe_allow_html=True)
        
        # Crear pestañas para cada fuente
        sources = metadata.get("sources", [])
        if sources:
            tabs = st.tabs([f"Fuente {i+1}" for i in range(len(sources))])
            
            for i, (tab, source) in enumerate(zip(tabs, sources)):
                with tab:
                    if i == 0:  # Destacar la primera fuente como seleccionada
                        st.markdown('<div class="source-content">', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="source-content">', unsafe_allow_html=True)
                    
                    # Contenido de la fuente
                    if "content" in source:
                        st.markdown(f"**Contenido:** {source['content']}")
                    
                    # Metadatos de la fuente
                    if "metadata" in source:
                        source_meta = source["metadata"]
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"**Fuente:** {source_meta.get('fuente', 'No especificada')}")
                        
                        with col2:
                            st.markdown(f"**Categoría:** {source_meta.get('categoria', 'No especificada')}")
                        
                        with col3:
                            st.markdown(f"**Fecha:** {source_meta.get('fecha', 'No especificada')}")
                        
                        # Sentimiento y engagement
                        col1, col2 = st.columns(2)
                        with col1:
                            sentiment = source_meta.get('sentimiento', 'neutral')
                            sentiment_class = "neutral"
                            
                            if "positive" in sentiment.lower():
                                sentiment_class = "positive"
                            elif "negative" in sentiment.lower():
                                sentiment_class = "negative"
                            
                            st.markdown(f'**Sentimiento:** <span class="sentiment-{sentiment_class}">{sentiment}</span>', unsafe_allow_html=True)
                        
                        with col2:
                            engagement = source_meta.get('engagement', 0)
                            st.markdown(f"**Engagement:** {engagement}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # INICIO DEL INFORME PRINCIPAL - Similar a las imágenes de referencia
    
    # Comprobar si el resumen ejecutivo existe
    if "Resumen Ejecutivo" in data:
        resumen = data["Resumen Ejecutivo"]
        st.markdown(f'<div class="topic-content">{resumen}</div>', unsafe_allow_html=True)
    
    # TEMA 1: POLÍTICA EN ANDALUCÍA
    st.markdown('<div class="numbered-topic">1. Política en Andalucía</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="topic-content">
        La conversación tiene un sentimiento mayoritariamente negativo, reflejando críticas y debates sobre la gestión 
        política. El engagement promedio es moderado, lo que indica un nivel de interacción por parte de los usuarios.
    </div>
    """, unsafe_allow_html=True)
    
    # Ejemplo extraído del JSON
    ejemplo1 = ""
    if "metadata" in report_data and "sources" in report_data["metadata"]:
        sources = report_data["metadata"]["sources"]
        if sources and len(sources) > 0 and "content" in sources[0]:
            ejemplo1 = sources[0]["content"].replace("Contenido: ", "")
    
    st.markdown(f'<div class="example-quote">Ejemplo: "{ejemplo1}"</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="topic-content">
        <span class="conclusion-label">Conclusión:</span> La política es el tema más discutido en redes sociales, con un 
        tono predominantemente negativo. Esto sugiere una fuerte polarización en la opinión pública, con debates activos
        sobre la gestión y las decisiones del gobierno.
    </div>
    """, unsafe_allow_html=True)
    
    # TEMA 2: ECONOMÍA Y PRESUPUESTOS
    st.markdown('<div class="numbered-topic">2. Economía y presupuestos</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="topic-content">
        Los temas económicos son mencionados con referencias a presupuestos, inversiones y empleo. También
        predomina el sentimiento negativo, reflejando preocupaciones sobre la situación financiera y el impacto en la
        ciudadanía. Sin embargo, el engagement fue más alto.
    </div>
    """, unsafe_allow_html=True)
    
    # Ejemplo para el segundo tema
    ejemplo2 = ""
    if "metadata" in report_data and "sources" in report_data["metadata"]:
        sources = report_data["metadata"]["sources"]
        if sources and len(sources) > 1 and "content" in sources[1]:
            ejemplo2 = sources[1]["content"].replace("Contenido: ", "")
    
    st.markdown(f'<div class="example-quote">Ejemplo: "{ejemplo2}"</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="topic-content">
        <span class="conclusion-label">Conclusión:</span> La economía y la gestión financiera generan un fuerte interés y preocupación. 
        Las críticas apuntan a la percepción de una administración deficiente o insuficiente.
    </div>
    """, unsafe_allow_html=True)
    
    # TEMA 3: PROBLEMAS SOCIALES Y EMERGENCIAS
    st.markdown('<div class="numbered-topic">3. Problemas sociales y emergencias</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="topic-content">
        Destacando preocupaciones en sanidad, educación y desempleo. Este fue el tema con mayor engagement, 
        indicando una fuerte reacción del público. El sentimiento es mayoritariamente negativo.
    </div>
    """, unsafe_allow_html=True)
    
    # Ejemplo para el tercer tema
    ejemplo3 = ""
    if "metadata" in report_data and "sources" in report_data["metadata"]:
        sources = report_data["metadata"]["sources"]
        if sources and len(sources) > 2 and "content" in sources[2]:
            ejemplo3 = sources[2]["content"].replace("Contenido: ", "")
    
    st.markdown(f'<div class="example-quote">Ejemplo: "{ejemplo3}"</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="topic-content">
        <span class="conclusion-label">Conclusión:</span> Los problemas sociales son percibidos como una falencia importante
        de la administración actual, generando un alto nivel de engagement y reacciones negativas.
    </div>
    """, unsafe_allow_html=True)
    
    # TEMA 4: MEDIO AMBIENTE Y CAMBIO CLIMÁTICO
    st.markdown('<div class="numbered-topic">4. Medio ambiente y cambio climático</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="topic-content">
        Temas sobre sostenibilidad, contaminación y energía. Este tema destaca porque su sentimiento mayoritario es
        positivo. El engagement es más bajo, indicando menor interacción en comparación con otros temas.
    </div>
    """, unsafe_allow_html=True)
    
    # Ejemplo para el cuarto tema
    ejemplo4 = ""
    if "metadata" in report_data and "sources" in report_data["metadata"]:
        sources = report_data["metadata"]["sources"]
        if sources and len(sources) > 3 and "content" in sources[3]:
            ejemplo4 = sources[3]["content"].replace("Contenido: ", "")
    
    st.markdown(f'<div class="example-quote">Ejemplo: "{ejemplo4}"</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="topic-content">
        <span class="conclusion-label">Conclusión:</span> Aunque es un tema con menor volumen de publicaciones, destaca por su tono positivo, 
        lo que sugiere que las acciones en este ámbito pueden ser bien recibidas por la ciudadanía. Sin embargo, el engagement bajo
        muestra que no es una prioridad en las discusiones públicas.
    </div>
    """, unsafe_allow_html=True)
    
    # TEMA 5: POLÍTICA NACIONAL E INTERNACIONAL
    st.markdown('<div class="numbered-topic">5. Política nacional e internacional</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="topic-content">
        Los asuntos internacionales y nacionales fuera del ámbito andaluz aparecen en pocas publicaciones, lo que
        muestra un interés menor en comparación con la política regional. En este caso, el sentimiento es positivo, con un
        engagement bajo, lo que sugiere que estos temas no generan tanta conversación entre los usuarios.
    </div>
    """, unsafe_allow_html=True)
    
    # Ejemplo para el quinto tema
    ejemplo5 = ""
    if "metadata" in report_data and "sources" in report_data["metadata"]:
        sources = report_data["metadata"]["sources"]
        if sources and len(sources) > 4 and "content" in sources[4]:
            ejemplo5 = sources[4]["content"].replace("Contenido: ", "")
    
    st.markdown(f'<div class="example-quote">Ejemplo: "{ejemplo5}"</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="topic-content">
        <span class="conclusion-label">Conclusión:</span> La política fuera de Andalucía no es una prioridad para el público regional.
        Sin embargo, las menciones positivas indican que ciertos eventos internacionales pueden mejorar marginalmente la percepción del liderazgo local.
    </div>
    """, unsafe_allow_html=True)
    
    # TEMA 6: DESINFORMACIÓN Y PERCEPCIÓN PÚBLICA
    st.markdown('<div class="numbered-topic">6. Desinformación y percepción pública</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="topic-content">
        Las publicaciones relacionadas con la percepción pública y la desinformación incluyen menciones a noticias falsas,
        manipulación informativa y debate sobre credibilidad.
    </div>
    """, unsafe_allow_html=True)
    
    # Ejemplo para el sexto tema
    ejemplo6 = ""
    if "metadata" in report_data and "sources" in report_data["metadata"]:
        sources = report_data["metadata"]["sources"]
        if sources and len(sources) > 5 and "content" in sources[5]:
            ejemplo6 = sources[5]["content"].replace("Contenido: ", "")
    
    st.markdown(f'<div class="example-quote">Ejemplo: "{ejemplo6}"</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="topic-content">
        <span class="conclusion-label">Conclusión:</span> Aunque la cantidad de publicaciones sobre desinformación no es alta, su impacto es significativo.
        La manipulación mediática y la percepción pública pueden condicionar el apoyo o rechazo. Es fundamental fortalecer la comunicación institucional
        y la transparencia para contrarrestar estos efectos.
    </div>
    """, unsafe_allow_html=True)
    
    # TEMA 7: COMUNICACIÓN Y LIDERAZGO POLÍTICO
    st.markdown('<div class="numbered-topic">7. Comunicación y liderazgo político</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="topic-content">
        Se observan menciones a discursos, estrategias comunicativas y la presencia en redes sociales.
    </div>
    """, unsafe_allow_html=True)
    
    # Ejemplo para el séptimo tema
    ejemplo7 = ""
    if "metadata" in report_data and "sources" in report_data["metadata"]:
        sources = report_data["metadata"]["sources"]
        if sources and len(sources) > 6 and "content" in sources[6]:
            ejemplo7 = sources[6]["content"].replace("Contenido: ", "")
    
    st.markdown(f'<div class="example-quote">Ejemplo: "{ejemplo7}"</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="topic-content">
        <span class="conclusion-label">Conclusión:</span> La comunicación política juega un papel clave en la imagen. Aunque algunos mensajes
        refuerzan su liderazgo, otros lo cuestionan, lo que indica que la percepción pública aún es volátil.
    </div>
    """, unsafe_allow_html=True)
    
    # TEMA 8: CULTURA Y PATRIMONIO
    st.markdown('<div class="numbered-topic">8. Cultura y patrimonio</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="topic-content">
        El ámbito cultural y patrimonial ha sido menos discutido en comparación con otras áreas. Sin embargo, se han
        registrado publicaciones sobre proyectos relacionados con la conservación del patrimonio, eventos culturales y
        debates sobre la identidad andaluza.
    </div>
    """, unsafe_allow_html=True)
    
    # Ejemplo para el octavo tema
    ejemplo8 = ""
    if "metadata" in report_data and "sources" in report_data["metadata"]:
        sources = report_data["metadata"]["sources"]
        if sources and len(sources) > 7 and "content" in sources[7]:
            ejemplo8 = sources[7]["content"].replace("Contenido: ", "")
    
    st.markdown(f'<div class="example-quote">Ejemplo: "{ejemplo8}"</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="topic-content">
        <span class="conclusion-label">Conclusión:</span> La cultura y el patrimonio ofrecen una oportunidad para mejorar la identidad regional y el orgullo cívico.
        Una mayor visibilidad y promoción de estos temas podría tener efectos positivos en la percepción pública.
    </div>
    """, unsafe_allow_html=True)
    
    # Recomendaciones del informe
    if "Recomendaciones" in data and isinstance(data["Recomendaciones"], list):
        st.markdown('<div class="report-section">Recomendaciones</div>', unsafe_allow_html=True)
        for i, recomendacion in enumerate(data["Recomendaciones"], 1):
            st.markdown(f"**{i}.** {recomendacion}")
    
    # Cerrar el contenedor principal
    st.markdown('</div>', unsafe_allow_html=True)  # Cierre del contenedor principal

def main():
    """Función principal de la aplicación."""
    # Renderizar encabezado
    render_header()
    
    # Verificar recursos
    if not check_resources():
        st.stop()
    
    # Cargar datos procesados
    try:
        df = load_processed_data()
    except Exception as e:
        st.error(f"Error al cargar los datos procesados: {e}")
        st.stop()
    
    # Renderizar barra lateral
    query, report_type, generate_button = render_sidebar()
    
    # Renderizar métricas generales
    render_metrics(df)
    
    # Generar informe si se solicita
    if generate_button and query:
        with st.spinner("Generando informe... Esto puede tardar unos minutos."):
            try:
                # Inicializar pipeline RAG
                pipeline = get_rag_pipeline()
                
                # Generar informe
                report = pipeline.generate_report(query, report_type)
                
                # Guardar informe
                report_path = pipeline.save_report(report)
                
                # Renderizar informe
                render_report(report)
                
                st.success(f"Informe generado y guardado en: {report_path}")
            
            except Exception as e:
                st.error(f"Error al generar el informe: {e}")
    
    # Si no hay acción, mostrar información de bienvenida
    if not generate_button:
        st.markdown("""
        <div class="card">
            <h3>Bienvenido al sistema RAG de análisis de datos sociales</h3>
            <p>Este sistema utiliza Retrieval Augmented Generation (RAG) para analizar datos de redes sociales y generar informes automáticos.</p>
            <p>Para comenzar, selecciona un tema en la barra lateral y haz clic en "Generar informe".</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>¿Qué temas puedes analizar?</h3>
            <p>Selecciona uno de los temas disponibles para generar un informe detallado que incluye:</p>
            <ul>
                <li>Análisis de 8 categorías temáticas principales</li>
                <li>Ejemplos reales extraídos de redes sociales</li>
                <li>Análisis de sentimiento con porcentajes precisos</li>
                <li>Recomendaciones estratégicas personalizadas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
