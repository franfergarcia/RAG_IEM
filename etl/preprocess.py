"""
Script para preprocesar los datos de redes sociales.
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
import re
import hashlib
import unicodedata
from dateutil import parser

# Añadir el directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cargar variables de entorno
load_dotenv()

def generate_unique_id(row):
    """
    Genera un ID único para cada registro basado en sus datos.
    
    Args:
        row (Series): Fila de datos.
        
    Returns:
        str: ID único.
    """
    # Concatenar valores relevantes
    values = []
    
    for field in ['FUENTE', 'TÍTULO', 'TEXTO', 'FECHA', 'USUARIO', 'NOMBRE DEL USUARIO']:
        if field in row and not pd.isna(row[field]):
            values.append(str(row[field]))
    
    # Si no hay suficientes datos para generar un ID único, usar timestamp
    if not values:
        values = [str(time.time())]
    
    # Generar hash
    text = '|'.join(values)
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def create_rich_text(row):
    """
    Crea un campo de texto enriquecido para RAG combinando varios campos.
    
    Args:
        row (Series): Fila de datos.
        
    Returns:
        str: Texto enriquecido.
    """
    parts = []
    
    # Añadir título si existe
    if 'TÍTULO' in row and not pd.isna(row['TÍTULO']):
        parts.append(f"Título: {row['TÍTULO']}")
    
    # Añadir texto principal
    if 'TEXTO' in row and not pd.isna(row['TEXTO']):
        parts.append(f"Contenido: {row['TEXTO']}")
    
    # Añadir descripción si existe y es diferente del texto
    if 'DESCRIPCIÓN' in row and not pd.isna(row['DESCRIPCIÓN']):
        if 'TEXTO' not in row or row['DESCRIPCIÓN'] != row['TEXTO']:
            parts.append(f"Descripción: {row['DESCRIPCIÓN']}")
    
    # Añadir metadatos relevantes
    metadata = []
    
    # Fuente
    if 'FUENTE' in row and not pd.isna(row['FUENTE']):
        metadata.append(f"Fuente: {row['FUENTE']}")
    
    # Categoría
    if 'CATEGORÍA' in row and not pd.isna(row['CATEGORÍA']):
        metadata.append(f"Categoría: {row['CATEGORÍA']}")
    
    # Búsqueda
    if 'BÚSQUEDA' in row and not pd.isna(row['BÚSQUEDA']):
        metadata.append(f"Búsqueda: {row['BÚSQUEDA']}")
    
    # Fecha
    if 'FECHA' in row and not pd.isna(row['FECHA']):
        metadata.append(f"Fecha: {row['FECHA']}")
    
    # Usuario
    if 'USUARIO_FINAL' in row and not pd.isna(row['USUARIO_FINAL']):
        metadata.append(f"Usuario: {row['USUARIO_FINAL']}")
    
    # Engagement
    if 'ENGAGEMENT' in row and not pd.isna(row['ENGAGEMENT']):
        metadata.append(f"Engagement: {row['ENGAGEMENT']}")
    
    # Sentimiento
    if 'SENTIMIENTO' in row and not pd.isna(row['SENTIMIENTO']):
        metadata.append(f"Sentimiento: {row['SENTIMIENTO']}")
    
    # Añadir metadatos al texto enriquecido
    if metadata:
        parts.append("Metadatos: " + "; ".join(metadata))
    
    # Unir todas las partes
    return "\n\n".join(parts)

def preprocess_data():
    """
    Preprocesa los datos de redes sociales.
    
    Returns:
        pd.DataFrame: DataFrame con los datos preprocesados.
    """
    # Obtener rutas de los archivos
    input_csv = os.getenv("INPUT_CSV", "Junta Andalucia datos.csv")
    
    # Obtener la ruta absoluta al directorio raíz del proyecto
    project_root = Path(__file__).parent.parent.absolute()
    input_path = os.path.join(project_root, "data", "raw", input_csv)
    
    processed_csv = os.getenv("PROCESSED_CSV", "processed_data.csv")
    output_path = os.path.join(project_root, "data", "processed", processed_csv)
    
    logger.info(f"Ruta de entrada: {input_path}")
    logger.info(f"Ruta de salida: {output_path}")
    
    # Verificar que exista el archivo de entrada
    if not os.path.exists(input_path):
        logger.error(f"No se encontró el archivo de entrada: {input_path}")
        return None
    
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Cargar datos
        logger.info(f"Cargando datos desde: {input_path}")
        df = pd.read_csv(input_path, delimiter=';', encoding='utf-8', low_memory=False)
        
        logger.info(f"Datos cargados: {len(df)} registros")
        
        # Seleccionar solo las columnas relevantes
        relevant_columns = [
            'FUENTE', 'CATEGORÍA', 'BÚSQUEDA', 'TÍTULO', 'TEXTO', 'DESCRIPCIÓN',
            'FECHA', 'COMPARTIDOS', 'ME GUSTAS', 'NO ME GUSTAS', 'COMENTARIOS',
            'REACCIONES', 'SENTIMIENTO', 'ENGAGEMENT', 'IMPRESIONES', 'INTERACCIONES',
            'ALCANCE', 'VALOR AYZENBERG', 'NOMBRE DEL USUARIO', 'USUARIO',
            'SEGUIDOS', 'SEGUIDORES', 'PUBLICACIONES'
        ]
        
        # Filtrar solo las columnas que existen en el DataFrame
        existing_columns = [col for col in relevant_columns if col in df.columns]
        df = df[existing_columns]
        
        logger.info(f"Columnas seleccionadas: {len(existing_columns)} de {len(relevant_columns)}")
        
        # Eliminar duplicados
        before_dedup = len(df)
        df = df.drop_duplicates()
        after_dedup = len(df)
        logger.info(f"Duplicados eliminados: {before_dedup - after_dedup} registros")
        
        # Gestionar campos de usuario
        logger.info("Procesando campos de usuario...")
        df['USUARIO_FINAL'] = df.apply(
            lambda row: row['USUARIO'] if 'USUARIO' in df.columns and not pd.isna(row['USUARIO']) 
            else row['NOMBRE DEL USUARIO'] if 'NOMBRE DEL USUARIO' in df.columns and not pd.isna(row['NOMBRE DEL USUARIO']) 
            else None, 
            axis=1
        )
        
        # Generar ID único
        logger.info("Generando IDs únicos...")
        df['ID_UNICO'] = df.apply(generate_unique_id, axis=1)
        
        # Crear texto enriquecido para RAG
        logger.info("Creando texto enriquecido para RAG...")
        df['TEXTO_ENRIQUECIDO'] = df.apply(create_rich_text, axis=1)
        
        # Guardar datos procesados
        logger.info(f"Guardando datos procesados en: {output_path}")
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.success(f"Datos preprocesados guardados: {len(df)} registros")
        
        # Estadísticas de los datos procesados
        stats = {
            "total_registros": len(df),
            "registros_con_texto": df['TEXTO'].notna().sum(),
            "registros_con_fecha": df['FECHA'].notna().sum(),
            "fuentes_unicas": df['FUENTE'].nunique() if 'FUENTE' in df.columns else 0,
            "categorias_unicas": df['CATEGORÍA'].nunique() if 'CATEGORÍA' in df.columns else 0
        }
        
        logger.info(f"Estadísticas de los datos procesados: {stats}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error al preprocesar los datos: {e}")
        return None

def main():
    """Función principal."""
    logger.info("Iniciando preprocesamiento de datos")
    
    # Verificar directorios
    project_root = Path(__file__).parent.parent.absolute()
    for directory in ["data/raw", "data/processed"]:
        os.makedirs(os.path.join(project_root, directory), exist_ok=True)
    
    # Preprocesar datos
    df = preprocess_data()
    
    if df is not None:
        logger.success("Proceso de preprocesamiento completado con éxito")
    else:
        logger.error("Error en el proceso de preprocesamiento")
        sys.exit(1)

if __name__ == "__main__":
    main()
