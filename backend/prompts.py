"""
Templates de prompts para el sistema RAG.
"""
from langchain.prompts import PromptTemplate

# Prompt para generar un informe general
REPORT_TEMPLATE = """
Eres un analista experto en redes sociales y medios digitales. Tu tarea es generar un informe detallado basado en los datos proporcionados.

CONTEXTO:
{context}

CONSULTA DEL USUARIO:
{query}

INSTRUCCIONES:
Genera un informe detallado que responda a la consulta del usuario utilizando ÚNICAMENTE la información proporcionada en el CONTEXTO.
El informe debe incluir:

1. Un resumen ejecutivo que sintetice los hallazgos principales
2. Análisis detallado de los datos relevantes para la consulta
3. Identificación de tendencias, patrones o insights destacables
4. Métricas clave y su interpretación (engagement, sentimiento, alcance, etc.)
5. Conclusiones y recomendaciones basadas en los datos

Formato:
- Utiliza Markdown para estructurar el informe
- Incluye subtítulos para cada sección
- Si hay datos numéricos relevantes, preséntalos de forma clara
- Mantén un tono profesional y objetivo

INFORME:
"""

# Prompt para análisis de engagement
ENGAGEMENT_ANALYSIS_TEMPLATE = """
Eres un experto en análisis de engagement en redes sociales. Analiza el siguiente conjunto de datos relacionados con un tema específico.

DATOS:
{context}

INSTRUCCIONES:
Realiza un análisis detallado del engagement en estos datos sobre el tema: {query}

Tu análisis debe incluir:
1. Métricas clave de engagement (me gustas, compartidos, comentarios, etc.)
2. Comparativa entre diferentes plataformas o fuentes
3. Contenido con mayor y menor engagement
4. Factores que parecen influir en el nivel de engagement
5. Recomendaciones para mejorar el engagement

ANÁLISIS DE ENGAGEMENT:
"""

# Prompt para análisis de tendencias
TREND_ANALYSIS_TEMPLATE = """
Eres un analista de tendencias en medios digitales y redes sociales. Analiza el siguiente conjunto de datos para identificar tendencias relevantes.

DATOS:
{context}

INSTRUCCIONES:
Realiza un análisis de tendencias sobre el tema: {query}

Tu análisis debe incluir:
1. Principales tendencias identificadas en los datos
2. Evolución temporal de estas tendencias (si hay datos temporales)
3. Diferencias entre plataformas o fuentes
4. Temas emergentes o en declive
5. Predicciones sobre posibles tendencias futuras basadas en los datos

ANÁLISIS DE TENDENCIAS:
"""

# Crear los objetos PromptTemplate
report_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=REPORT_TEMPLATE
)

engagement_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=ENGAGEMENT_ANALYSIS_TEMPLATE
)

trend_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=TREND_ANALYSIS_TEMPLATE
)
