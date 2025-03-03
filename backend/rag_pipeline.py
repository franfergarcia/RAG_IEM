"""
Pipeline RAG para el análisis de datos sociales.
"""
import os
import sys
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
from loguru import logger
import torch
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import pickle
import shutil
import types

# Cargar variables de entorno
load_dotenv()

class RAGPipeline:
    """
    Pipeline de Retrieval Augmented Generation para análisis de datos sociales.
    """
    
    def __init__(self):
        """Inicializa el pipeline RAG."""
        self.processed_data_path = os.getenv("PROCESSED_DATA_PATH", "data/processed/processed_data.csv")
        self.vectorstore_path = os.getenv("VECTORSTORE_PATH", "data/vectorstore")
        self.llm_model = os.getenv("LLM_MODEL", "granite3.2")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "granite-embedding:278m")
        self.top_k = int(os.getenv("TOP_K_RESULTS", "5"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Componentes del pipeline
        self.data = None
        self.vectorstore = None
        self.llm = None
        
        logger.info(f"Pipeline RAG inicializado con modelo LLM: {self.llm_model}, modelo de embedding: {self.embedding_model}")
    
    def initialize(self):
        """Inicializa todos los componentes del pipeline."""
        try:
            # Cargar datos procesados
            self._load_data()
            
            # Cargar vectorstore
            self._load_vectorstore()
            
            # Inicializar LLM
            self._initialize_llm()
            
            # Inicializar generador de informes
            self._initialize_report_generator()
            
            logger.success("Pipeline RAG inicializado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar el pipeline RAG: {e}")
            return False
    
    def _load_data(self):
        """Carga los datos procesados."""
        try:
            if not os.path.exists(self.processed_data_path):
                raise FileNotFoundError(f"No se encontró el archivo de datos procesados: {self.processed_data_path}")
            
            self.data = pd.read_csv(self.processed_data_path)
            logger.info(f"Datos cargados correctamente: {len(self.data)} registros")
            return True
        except Exception as e:
            logger.error(f"Error al cargar los datos: {e}")
            return False
    
    def _load_vectorstore(self):
        """
        Carga el vectorstore FAISS.
        
        Returns:
            bool: True si se cargó correctamente, False en caso contrario.
        """
        try:
            # Verificar que exista el directorio del vectorstore
            if not os.path.exists(self.vectorstore_path):
                raise FileNotFoundError(f"No se encontró el directorio del vectorstore: {self.vectorstore_path}")
            
            # Cargar el modelo de embeddings
            logger.info(f"Cargando modelo de embeddings: {self.embedding_model}")
            
            # Verificar disponibilidad de GPU
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.warning("No se detectó GPU, usando CPU para los embeddings")
            
            # Corregir el formato del modelo para Hugging Face si es necesario
            embedding_model = self.embedding_model
            if ":" in embedding_model:
                embedding_model = "ibm-granite/granite-embedding-278m-multilingual"
            
            # Inicializar el modelo de embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': device}
            )
            
            # Cargar el vectorstore FAISS
            logger.info(f"Cargando vectorstore FAISS desde: {self.vectorstore_path}")
            
            # Verificar que existan los archivos necesarios
            index_path = os.path.join(self.vectorstore_path, "index.faiss")
            pkl_path = os.path.join(self.vectorstore_path, "index.pkl")
            
            if not os.path.exists(index_path) or not os.path.exists(pkl_path):
                raise FileNotFoundError(f"No se encontraron los archivos necesarios del vectorstore en: {self.vectorstore_path}")
            
            # Cargar el vectorstore utilizando el método estándar de LangChain
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Verificar que se haya cargado correctamente
            doc_count = self.vectorstore.index.ntotal
            logger.success(f"Vectorstore FAISS cargado correctamente con {doc_count} documentos")
            return True
            
        except Exception as e:
            logger.error(f"Error al cargar el vectorstore con FAISS: {e}")
            logger.error("Por favor, regenere el vectorstore ejecutando: python main.py etl --step vectorstore")
            return False
    
    def _initialize_llm(self):
        """Inicializa el LLM."""
        try:
            # Inicializar el modelo LLM
            logger.info(f"Inicializando LLM con modelo: {self.llm_model}")
            
            # Inicializar Ollama con el modelo especificado
            self.llm = Ollama(model=self.llm_model)
            
            logger.success(f"LLM inicializado correctamente con modelo: {self.llm_model}")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar el LLM: {e}")
            return False
    
    def _initialize_report_generator(self):
        """Inicializa el generador de informes."""
        try:
            # Crear plantillas de prompts para diferentes tipos de informes
            self.prompts = {
                "general": PromptTemplate(
                    template="""Eres un analista experto en redes sociales y marketing digital. 
                    Genera un informe completo y detallado sobre el siguiente tema: {query}
                    
                    Utiliza los siguientes datos como contexto para tu análisis:
                    {context}
                    
                    El informe DEBE incluir EXACTAMENTE los siguientes 8 temas numerados:
                    1. Política en Andalucía
                    2. Economía y presupuestos
                    3. Problemas sociales y emergencias
                    4. Medio ambiente y cambio climático
                    5. Política nacional e internacional
                    6. Desinformación y percepción pública
                    7. Comunicación y liderazgo político
                    8. Cultura y patrimonio
                    
                    Para CADA uno de los 8 temas, incluye:
                    - Una descripción detallada del tema (con datos de engagement si es relevante)
                    - Al menos un ejemplo real con cita textual
                    - Una conclusión clara que resuma los hallazgos principales
                    
                    Además, el informe debe incluir:
                    1. Resumen ejecutivo general
                    2. Análisis de tendencias principales
                    3. Análisis de sentimiento general con porcentajes precisos
                    4. Análisis de engagement por tema
                    5. Al menos 5 recomendaciones concretas y procesables
                    
                    Formato en JSON estructurado con las secciones mencionadas.
                    
                    NOTA: Es IMPERATIVO que incluyas EXACTAMENTE los 8 temas listados anteriormente, en ese orden.
                    """,
                    input_variables=["query", "context"]
                ),
                "sentiment": PromptTemplate(
                    template="""Eres un analista experto en análisis de sentimiento en redes sociales.
                    Genera un informe detallado de sentimiento sobre: {query}
                    
                    Utiliza los siguientes datos como contexto para tu análisis:
                    {context}
                    
                    El informe DEBE incluir:
                    1. Resumen ejecutivo del sentimiento general
                    2. Distribución de sentimiento:
                       - IMPORTANTE: Incluye porcentajes precisos para cada categoría (Positivo, Neutral, Negativo)
                       - Estos porcentajes DEBEN sumar EXACTAMENTE 100%
                       - Expresa los porcentajes con un decimal (por ejemplo: 45.5%)
                    3. Análisis de los factores que influyen en cada tipo de sentimiento
                    4. Evolución del sentimiento en el tiempo
                    5. Recomendaciones para mejorar el sentimiento
                    
                    Formato en JSON estructurado con las secciones mencionadas.
                    
                    NOTA: Es CRUCIAL que los porcentajes de sentimiento Positivo, Neutral y Negativo sumen EXACTAMENTE 100%.
                    """,
                    input_variables=["query", "context"]
                ),
                "engagement": PromptTemplate(
                    template="""Eres un analista experto en engagement en redes sociales.
                    Genera un informe detallado de engagement sobre: {query}
                    
                    Utiliza los siguientes datos como contexto para tu análisis:
                    {context}
                    
                    El informe debe incluir:
                    1. Resumen ejecutivo del engagement general
                    2. Métricas clave de engagement (likes, comentarios, compartidos)
                    3. Análisis de los factores que influyen en el engagement
                    4. Comparativa entre plataformas
                    5. Recomendaciones para mejorar el engagement
                    
                    Formato en JSON estructurado con las secciones mencionadas.
                    """,
                    input_variables=["query", "context"]
                )
            }
            
            logger.success(f"Generador de informes inicializado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar el generador de informes: {e}")
            return False
    
    def search_documents(self, query, top_k=None):
        """
        Busca documentos relevantes para una consulta utilizando el vectorstore.
        
        Args:
            query (str): Consulta para la búsqueda.
            top_k (int, optional): Número de documentos a recuperar. 
                                Si es None, se usa self.top_k.
        
        Returns:
            list: Lista de documentos relevantes.
        """
        try:
            if self.vectorstore is None:
                raise ValueError("El vectorstore no está inicializado")
            
            # Usar top_k de clase si no se especifica
            if top_k is None:
                top_k = self.top_k
            
            # Buscar documentos
            logger.info(f"Buscando documentos para: '{query}'")
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
            
            # Extraer documentos y formatear resultados
            results = []
            for i, (doc, score) in enumerate(docs_with_scores):
                # Convertir score a una similitud normalizada (0-1)
                similarity = 1.0 - min(score, 1.0)  # Los scores de FAISS son distancias (menor es mejor)
                
                # Agregar a resultados
                results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": similarity,
                    "rank": i + 1
                })
            
            logger.info(f"Encontrados {len(results)} documentos relevantes")
            return results
            
        except Exception as e:
            logger.error(f"Error al buscar documentos: {e}")
            logger.error("Por favor, regenere el vectorstore ejecutando: python main.py etl --step vectorstore")
            return []
    
    def generate_report(self, query, report_type="general"):
        """
        Genera un informe basado en la consulta y el tipo de informe.
        
        Args:
            query (str): Consulta para el informe.
            report_type (str): Tipo de informe ('general', 'sentiment', 'engagement').
            
        Returns:
            dict: Informe generado.
        """
        start_time = time.time()
        
        try:
            if not query:
                return {
                    "error": "La consulta no puede estar vacía",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Inicializar el pipeline si es necesario
            if self.data is None or self.vectorstore is None or self.llm is None:
                self.initialize()
            
            logger.info(f"Generando informe de tipo '{report_type}' para consulta: '{query}'")
            
            # 1. Buscar documentos relevantes en el vectorstore
            logger.info(f"Buscando documentos relevantes para: '{query}'")
            relevant_docs = self.search_documents(query)
            
            # Verificar si se encontraron documentos
            if not relevant_docs:
                error_msg = "No se pudieron encontrar documentos relevantes"
                logger.warning(error_msg)
                return {
                    "error": error_msg,
                    "query": query,
                    "type": report_type,
                    "timestamp": datetime.now().isoformat()
                }
            
            # 2. Preparar el contexto para el LLM
            context = "\n\n".join([doc["text"] for doc in relevant_docs])
            
            # 3. Seleccionar la plantilla de prompt adecuada
            if report_type not in self.prompts:
                logger.warning(f"Tipo de informe '{report_type}' no reconocido, usando 'general'")
                report_type = "general"
            
            prompt_template = self.prompts[report_type]
            
            # 4. Generar el informe directamente sin usar RetrievalQA
            formatted_prompt = prompt_template.format(context=context, query=query)
            
            # 5. Generar el informe
            logger.info(f"Generando informe con el LLM usando la plantilla de '{report_type}'")
            result = self.llm.invoke(formatted_prompt)
            
            # 6. Procesar el resultado
            try:
                # Intentar extraer el JSON si está en formato de cadena
                if isinstance(result, dict) and "text" in result:
                    result_text = result["text"]
                else:
                    result_text = str(result)
                
                # Intentar extraer JSON si está en formato de texto
                try:
                    # Buscar el inicio y fin del JSON en el texto
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = result_text[start_idx:end_idx]
                        report_data = json.loads(json_str)
                    else:
                        report_data = {"content": result_text}
                except json.JSONDecodeError:
                    report_data = {"content": result_text}
            except Exception as e:
                logger.warning(f"Error al procesar el resultado como JSON: {e}")
                report_data = {"content": str(result)}
            
            # 7. Añadir metadatos al informe
            report = {
                "query": query,
                "type": report_type,
                "timestamp": datetime.now().isoformat(),
                "data": report_data,
                "metadata": {
                    "model": self.llm_model,
                    "embedding_model": self.embedding_model,
                    "documents_retrieved": len(relevant_docs),
                    "sources": [
                        {
                            "content": doc["text"],  # Mostrar contenido completo
                            "metadata": doc["metadata"]
                        } for doc in relevant_docs
                    ]
                }
            }
            
            logger.success(f"Informe generado correctamente")
            
            end_time = time.time()
            report["generation_time"] = end_time - start_time
            
            logger.success(f"Informe generado correctamente en {report['generation_time']:.2f} segundos")
            return report
        
        except Exception as e:
            logger.error(f"Error al generar el informe: {e}")
            return {"error": str(e), "query": query, "type": report_type}
    
    def save_report(self, report):
        """
        Guarda el informe generado.
        
        Args:
            report (dict): Informe a guardar.
            
        Returns:
            str: Ruta donde se guardó el informe.
        """
        try:
            # Crear directorio de informes si no existe
            reports_dir = Path("data/reports")
            reports_dir.mkdir(exist_ok=True, parents=True)
            
            # Generar nombre de archivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_type = report.get("type", "general")
            filename = f"report_{report_type}_{timestamp}.json"
            report_path = reports_dir / filename
            
            # Guardar informe
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Informe guardado en: {report_path}")
            return str(report_path)
        
        except Exception as e:
            logger.error(f"Error al guardar el informe: {e}")
            return None
    
def main():
    """Función principal para probar el pipeline RAG."""
    # Ejemplo de uso
    pipeline = RAGPipeline()
    
    # Verificar recursos
    if pipeline.initialize():
        # Generar un informe de prueba
        query = "Opinión pública sobre la gestión política en Andalucía"
        report = pipeline.generate_report(query)
        
        # Guardar el informe
        if "error" not in report:
            filepath = pipeline.save_report(report)
            print(f"Informe guardado en: {filepath}")

if __name__ == "__main__":
    main()
