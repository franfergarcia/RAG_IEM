"""
Script para crear el vectorstore a partir de los datos procesados.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm
import json
import time
import random
import faiss
import torch
from langchain_community.vectorstores.faiss import FAISS as LangchainFAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
import importlib

# Añadir el directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cargar variables de entorno
load_dotenv()

def convert_to_float(value):
    """
    Convierte un valor a float, manejando diferentes formatos numéricos.
    
    Args:
        value: Valor a convertir
        
    Returns:
        float: Valor convertido a float o 0.0 si no se puede convertir
    """
    if pd.isna(value):
        return 0.0
    
    try:
        # Si ya es un número, devolverlo
        if isinstance(value, (int, float)):
            return float(value)
        
        # Convertir string a float
        if isinstance(value, str):
            # Reemplazar coma por punto para manejar formato europeo
            value = value.replace(',', '.')
            return float(value)
        
        return 0.0
    except:
        return 0.0

def create_vectorstore():
    """
    Crea el vectorstore a partir de los datos procesados usando FAISS y
    Ollama para generar embeddings.
    """
    # Obtener rutas de los archivos
    processed_csv = os.getenv("PROCESSED_CSV", "processed_data.csv")
    
    # Obtener la ruta absoluta al directorio raíz del proyecto
    project_root = Path(__file__).parent.parent.absolute()
    processed_path = os.path.join(project_root, "data", "processed", processed_csv)
    vectorstore_dir = os.path.join(project_root, "data", "vectorstore")
    temp_dir = os.path.join(project_root, "data", "temp")
    
    logger.info(f"Ruta de datos procesados: {processed_path}")
    logger.info(f"Directorio de vectorstore: {vectorstore_dir}")
    
    # Verificar que exista el archivo de entrada
    if not os.path.exists(processed_path):
        logger.error(f"No se encontró el archivo de datos procesados: {processed_path}")
        return False
    
    # Crear directorios si no existen
    os.makedirs(vectorstore_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Archivo para guardar el progreso
    progress_file = os.path.join(temp_dir, "vectorstore_progress.json")
    
    try:
        # Cargar datos procesados
        logger.info(f"Cargando datos procesados desde: {processed_path}")
        df = pd.read_csv(processed_path, encoding='utf-8')
        
        logger.info(f"Datos cargados: {len(df)} registros")
        
        # Verificar que exista la columna de texto enriquecido
        if 'TEXTO_ENRIQUECIDO' not in df.columns:
            logger.error("No se encontró la columna 'TEXTO_ENRIQUECIDO' en los datos procesados")
            return False
        
        # Obtener el modelo de embeddings desde las variables de entorno
        embedding_model = os.getenv("EMBEDDING_MODEL")
        
        if not embedding_model:
            logger.error("No se encontró la variable de entorno EMBEDDING_MODEL en el archivo .env")
            return False
            
        logger.info(f"Cargando modelo de embeddings {embedding_model} usando Ollama...")
        
        # Inicializar el modelo de embeddings con Ollama
        embeddings_model = OllamaEmbeddings(model=embedding_model)
        logger.info(f"Modelo de embeddings cargado correctamente con Ollama")
        
        # Crear documentos para FAISS
        logger.info("Creando documentos para FAISS...")
        documents = []
        
        # Procesar cada registro
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparando documentos"):
            # Crear metadatos
            metadata = {
                "id": row.get('ID_UNICO', f"doc_{idx}"),
                "fuente": row.get('FUENTE', ''),
                "categoria": row.get('CATEGORÍA', ''),
                "fecha": row.get('FECHA', ''),
                "sentimiento": row.get('SENTIMIENTO', ''),
                "engagement": convert_to_float(row.get('ENGAGEMENT', 0)),
                "usuario": row.get('USUARIO_FINAL', '')
            }
            
            # Crear documento
            doc = Document(
                page_content=row['TEXTO_ENRIQUECIDO'],
                metadata=metadata
            )
            
            documents.append(doc)
        
        logger.info(f"Preparados {len(documents)} documentos")
        
        # Verificar si hay progreso guardado
        start_index = 0
        embeddings_dict = {}
        
        if os.path.exists(progress_file):
            logger.info(f"Cargando progreso guardado desde: {progress_file}")
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                start_index = progress_data.get('next_index', 0)
                embeddings_dict = progress_data.get('embeddings', {})
            
            logger.info(f"Progreso cargado: {len(embeddings_dict)} documentos procesados, continuando desde el índice {start_index}")
        
        # Procesar documentos en lotes para mayor eficiencia
        logger.info("Generando embeddings para los documentos...")
        
        # Tamaño del lote para procesamiento eficiente
        batch_size = 32
        
        # Guardar progreso cada N lotes
        save_frequency = 5
        
        # Procesar documentos en lotes
        for i in tqdm(range(start_index, len(documents), batch_size), desc="Procesando lotes"):
            # Obtener lote actual
            end_idx = min(i + batch_size, len(documents))
            batch = documents[i:end_idx]
            
            # Extraer textos del lote
            texts = [doc.page_content for doc in batch]
            
            try:
                # Generar embeddings para todo el lote a la vez usando Ollama
                embeddings_batch = [embeddings_model.embed_query(text) for text in tqdm(texts, desc="Generando embeddings", leave=False)]
                
                # Guardar embeddings
                for j, doc in enumerate(batch):
                    doc_id = doc.metadata['id']
                    embeddings_dict[doc_id] = {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "embedding": embeddings_batch[j]
                    }
                
                # Guardar progreso periódicamente
                batch_num = (i // batch_size)
                if batch_num % save_frequency == 0 or end_idx >= len(documents):
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "next_index": end_idx,
                            "embeddings": embeddings_dict
                        }, f)
                    
                    logger.debug(f"Progreso guardado: {len(embeddings_dict)}/{len(documents)} documentos procesados")
            
            except Exception as e:
                logger.error(f"Error al procesar lote {i//batch_size + 1}: {e}")
                
                # Guardar progreso antes de continuar
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "next_index": i,
                        "embeddings": embeddings_dict
                    }, f)
                
                # Esperar un poco antes de continuar
                time.sleep(1)
        
        # Crear un nuevo vectorstore
        logger.info(f"Creando vectorstore FAISS con {len(documents)} documentos...")
        
        if not embeddings_dict:
            logger.error("No se pudieron generar embeddings para ningún documento")
            return False
        
        # Preparar datos para FAISS
        texts = []
        metadatas = []
        embeddings_list = []
        
        # Extraer datos del diccionario de embeddings
        for doc_id, data in embeddings_dict.items():
            texts.append(data["text"])
            metadatas.append(data["metadata"])
            embeddings_list.append(data["embedding"])
        
        # Verificar que todos los embeddings tengan la misma dimensión
        if not embeddings_list:
            logger.error("No hay embeddings válidos para crear el vectorstore")
            return False
        
        embedding_dim = len(embeddings_list[0])
        logger.info(f"Dimensión de los embeddings: {embedding_dim}")
        
        # Crear matriz de embeddings
        logger.info(f"Creando matriz de embeddings con {len(embeddings_list)} vectores")
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        
        # Crear índice FAISS
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings_array)
        
        # Crear objeto embeddings para LangChain - aquí usamos Ollama
        hf_embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Crear vectorstore apropiadamente utilizando la API moderna de LangChain
        # Creamos los documentos y metadatos que LangChain espera
        langchain_documents = []
        for text, metadata in zip(texts, metadatas):
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            langchain_documents.append(doc)
        
        # Crear el vectorstore utilizando el método estándar from_documents
        vectorstore = LangchainFAISS.from_documents(
            documents=langchain_documents,
            embedding=hf_embeddings
        )
        
        # Verificar que el vectorstore se creó correctamente
        doc_count = vectorstore.index.ntotal
        logger.info(f"Vectorstore creado correctamente con {doc_count} documentos")
        
        # Guardar el vectorstore
        logger.info(f"Guardando vectorstore en: {vectorstore_dir}")
        vectorstore.save_local(vectorstore_dir)
        
        # Guardar metadatos adicionales
        metadata_path = os.path.join(vectorstore_dir, "metadata.json")
        metadata = {
            "model": embedding_model,
            "document_count": len(documents),
            "embedding_dimension": len(embeddings_list[0]),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "columns": list(df.columns),
            "vectorstore_version": "1.0",  # Añadir versión para futuras compatibilidades
            "langchain_version": importlib.metadata.version("langchain")
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.success(f"Vectorstore FAISS creado y guardado en: {vectorstore_dir}")
        logger.info(f"Metadatos guardados en: {metadata_path}")
        
        # Eliminar archivo de progreso si todo fue exitoso
        if os.path.exists(progress_file):
            os.remove(progress_file)
        
        return True
    
    except Exception as e:
        logger.error(f"Error al crear el vectorstore: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Función principal."""
    logger.info("Iniciando creación del vectorstore")
    
    # Verificar directorios
    project_root = Path(__file__).parent.parent.absolute()
    for directory in ["data/processed", "data/vectorstore", "data/temp"]:
        os.makedirs(os.path.join(project_root, directory), exist_ok=True)
    
    # Crear vectorstore
    success = create_vectorstore()
    
    if success:
        logger.success("Vectorstore creado con éxito")
    else:
        logger.error("Error en la creación del vectorstore")
        sys.exit(1)

if __name__ == "__main__":
    main()
