"""
Generador de informes basado en el sistema RAG.
"""
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from loguru import logger
import time
from typing import Dict, List, Any, Optional, Union
import json
import re
from datetime import datetime
import sys

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, Ollama

from backend.prompts import report_prompt, sentiment_prompt, engagement_prompt
from backend.llm import get_llm

# Cargar variables de entorno
load_dotenv()

# Configurar logger
logger.add(sys.stderr)

class ReportGenerator:
    """Clase para generar informes basados en RAG."""
    
    def __init__(self, vectorstore_path: str = None, llm_model: str = None, embedding_model: str = None):
        """
        Inicializa el generador de informes.
        
        Args:
            vectorstore_path (str): Ruta al índice vectorial
            llm_model (str): Nombre del modelo LLM
            embedding_model (str): Nombre del modelo de embeddings
        """
        # Usar valores de variables de entorno si no se especifican
        self.vectorstore_path = vectorstore_path or os.getenv("VECTORSTORE_PATH")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "granite3.2")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "granite-embedding:278m")
        
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        
        logger.info(f"Inicializando ReportGenerator con modelo LLM: {self.llm_model}, embeddings: {self.embedding_model}")
    
    def load_resources(self):
        """Carga los recursos necesarios (LLM, embeddings, vectorstore)."""
        try:
            # Cargar LLM
            logger.info(f"Cargando LLM: {self.llm_model}")
            self.llm = get_llm(self.llm_model)
            
            # Cargar embeddings
            logger.info(f"Cargando modelo de embeddings: {self.embedding_model}")
            self.embeddings = OllamaEmbeddings(model=self.embedding_model)
            
            # Cargar vectorstore
            if os.path.exists(self.vectorstore_path):
                logger.info(f"Cargando vectorstore desde: {self.vectorstore_path}")
                self.vectorstore = FAISS.load_local(self.vectorstore_path, self.embeddings)
                logger.success("Vectorstore cargado correctamente")
            else:
                logger.error(f"No se encontró el vectorstore en: {self.vectorstore_path}")
                raise FileNotFoundError(f"No se encontró el vectorstore en: {self.vectorstore_path}")
            
            # Crear cadena de QA
            logger.info("Configurando cadena de QA")
            try:
                # Intentar usar as_retriever con el vectorstore
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": int(os.getenv("TOP_K_RESULTS", "5"))}
                )
            except Exception as e:
                logger.warning(f"Error al usar as_retriever: {e}")
                # Alternativa: crear un retriever personalizado que use similarity_search directamente
                from langchain.schema import BaseRetriever
                from langchain.schema.document import Document
                from typing import List

                class CustomRetriever(BaseRetriever):
                    """Retriever personalizado que usa similarity_search directamente."""
                    
                    def __init__(self, vectorstore, k=5):
                        self.vectorstore = vectorstore
                        self.k = k
                    
                    def get_relevant_documents(self, query: str) -> List[Document]:
                        """Obtiene documentos relevantes usando similarity_search."""
                        try:
                            return self.vectorstore.similarity_search(query, k=self.k)
                        except Exception:
                            # Si similarity_search falla, intentar con similarity_search_with_score
                            try:
                                docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.k)
                                return [doc for doc, _ in docs_with_scores]
                            except Exception as e:
                                logger.error(f"Error en retriever personalizado: {e}")
                                return []
                
                # Usar el retriever personalizado
                retriever = CustomRetriever(
                    vectorstore=self.vectorstore,
                    k=int(os.getenv("TOP_K_RESULTS", "5"))
                )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever
            )
            
            logger.success("Recursos cargados correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al cargar recursos: {e}")
            return False
    
    def generate_report(self, query: str, report_type: str = "general") -> Dict[str, Any]:
        """
        Genera un informe basado en una consulta.
        
        Args:
            query (str): Consulta o tema para el informe
            report_type (str): Tipo de informe (general, sentiment, engagement)
            
        Returns:
            Dict[str, Any]: Informe generado con metadatos
        """
        if not self.qa_chain:
            if not self.load_resources():
                return {"error": "No se pudieron cargar los recursos necesarios"}
        
        start_time = time.time()
        logger.info(f"Generando informe de tipo '{report_type}' para la consulta: '{query}'")
        
        try:
            # Seleccionar el prompt adecuado según el tipo de informe
            if report_type == "sentiment":
                prompt_template = sentiment_prompt
            elif report_type == "engagement":
                prompt_template = engagement_prompt
            else:  # general por defecto
                prompt_template = report_prompt
            
            # Configurar la cadena de QA con el prompt adecuado
            self.qa_chain.combine_documents_chain.llm_chain.prompt = prompt_template
            
            # Generar el informe
            result = self.qa_chain.invoke({"query": query})
            
            # Procesar y formatear el resultado
            report_content = result.get("result", "")
            
            # Crear el objeto de informe con metadatos
            report = {
                "query": query,
                "type": report_type,
                "content": report_content,
                "timestamp": datetime.now().isoformat(),
                "generation_time": time.time() - start_time,
                "model": self.llm_model
            }
            
            logger.success(f"Informe generado correctamente en {report['generation_time']:.2f} segundos")
            return report
        except Exception as e:
            logger.error(f"Error al generar el informe: {e}")
            return {"error": str(e)}
    
    def save_report(self, report: Dict[str, Any], output_dir: str = "data/reports") -> str:
        """
        Guarda un informe en disco.
        
        Args:
            report (Dict[str, Any]): Informe a guardar
            output_dir (str): Directorio donde guardar el informe
            
        Returns:
            str: Ruta al archivo guardado
        """
        try:
            # Crear directorio si no existe
            os.makedirs(output_dir, exist_ok=True)
            
            # Generar nombre de archivo
            query_slug = re.sub(r'[^\w\s]', '', report["query"]).strip().lower().replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{query_slug}_{report['type']}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Guardar como JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Informe guardado en: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error al guardar el informe: {e}")
            return ""

def main():
    """Función principal para probar el generador de informes."""
    # Ejemplo de uso
    generator = ReportGenerator()
    if generator.load_resources():
        # Generar un informe de prueba
        query = "Opinión pública sobre la gestión política en Andalucía"
        report = generator.generate_report(query)
        
        # Guardar el informe
        if "error" not in report:
            generator.save_report(report)

if __name__ == "__main__":
    main()
