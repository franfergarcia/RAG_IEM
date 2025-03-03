"""
Script para explorar y analizar el CSV de datos de redes sociales.
"""
import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from loguru import logger
import matplotlib.pyplot as plt
from pathlib import Path

# Añadir el directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cargar variables de entorno
load_dotenv()

def explore_csv():
    """
    Explora el CSV y muestra información sobre sus columnas y datos.
    """
    # Obtener la ruta del archivo CSV
    input_csv = os.getenv("INPUT_CSV", "Junta Andalucia datos.csv")
    
    # Obtener la ruta absoluta al directorio raíz del proyecto
    project_root = Path(__file__).parent.parent.absolute()
    input_path = os.path.join(project_root, "data", "raw", input_csv)
    
    logger.info(f"Buscando archivo en: {input_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"No se encontró el archivo: {input_path}")
        return
    
    logger.info(f"Explorando el archivo: {input_path}")
    
    try:
        # Cargar el CSV
        df = pd.read_csv(input_path, delimiter=';', encoding='utf-8', low_memory=False)
        
        # Información básica
        logger.info(f"Dimensiones del DataFrame: {df.shape}")
        logger.info(f"Número de filas: {len(df)}")
        logger.info(f"Número de columnas: {len(df.columns)}")
        
        # Mostrar todas las columnas
        logger.info("Columnas en el DataFrame:")
        for i, col in enumerate(df.columns, 1):
            logger.info(f"{i}. {col}")
        
        # Verificar columnas relevantes
        relevant_columns = [
            'FUENTE', 'CATEGORÍA', 'BÚSQUEDA', 'TÍTULO', 'TEXTO', 'DESCRIPCIÓN',
            'FECHA', 'COMPARTIDOS', 'ME GUSTAS', 'NO ME GUSTAS', 'COMENTARIOS',
            'REACCIONES', 'SENTIMIENTO', 'ENGAGEMENT', 'IMPRESIONES', 'INTERACCIONES',
            'ALCANCE', 'VALOR AYZENBERG', 'NOMBRE DEL USUARIO', 'USUARIO',
            'SEGUIDOS', 'SEGUIDORES', 'PUBLICACIONES'
        ]
        
        logger.info("Verificando columnas relevantes:")
        for col in relevant_columns:
            if col in df.columns:
                non_null = df[col].notna().sum()
                percentage = (non_null / len(df)) * 100
                logger.info(f"✓ {col}: {non_null} valores no nulos ({percentage:.2f}%)")
            else:
                logger.warning(f"✗ {col}: No encontrada en el DataFrame")
        
        # Verificar duplicados
        duplicates = df.duplicated().sum()
        logger.info(f"Filas duplicadas: {duplicates} ({(duplicates/len(df))*100:.2f}%)")
        
        # Distribución por fuente
        if 'FUENTE' in df.columns:
            logger.info("Distribución por FUENTE:")
            source_counts = df['FUENTE'].value_counts()
            for source, count in source_counts.items():
                logger.info(f"  - {source}: {count} ({(count/len(df))*100:.2f}%)")
        
        # Distribución por sentimiento
        if 'SENTIMIENTO' in df.columns:
            logger.info("Distribución por SENTIMIENTO:")
            sentiment_counts = df['SENTIMIENTO'].value_counts()
            for sentiment, count in sentiment_counts.items():
                logger.info(f"  - {sentiment}: {count} ({(count/len(df))*100:.2f}%)")
        
        # Distribución por categoría
        if 'CATEGORÍA' in df.columns:
            logger.info("Distribución por CATEGORÍA:")
            category_counts = df['CATEGORÍA'].value_counts().head(10)  # Top 10
            for category, count in category_counts.items():
                logger.info(f"  - {category}: {count} ({(count/len(df))*100:.2f}%)")
        
        # Estadísticas de engagement
        engagement_cols = ['COMPARTIDOS', 'ME GUSTAS', 'NO ME GUSTAS', 'COMENTARIOS', 
                          'REACCIONES', 'ENGAGEMENT', 'IMPRESIONES', 'INTERACCIONES']
        
        logger.info("Estadísticas de engagement:")
        for col in engagement_cols:
            if col in df.columns:
                # Convertir a numérico si es posible
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                stats = df[col].describe()
                logger.info(f"  - {col}:")
                logger.info(f"    - Media: {stats['mean']:.2f}")
                logger.info(f"    - Mediana: {stats['50%']:.2f}")
                logger.info(f"    - Máximo: {stats['max']:.2f}")
                logger.info(f"    - Valores no nulos: {df[col].notna().sum()} ({(df[col].notna().sum()/len(df))*100:.2f}%)")
        
        # Verificar campos de usuario
        user_cols = ['NOMBRE DE USUARIO', 'USUARIO']
        logger.info("Verificando campos de usuario:")
        
        for col in user_cols:
            if col in df.columns:
                non_null = df[col].notna().sum()
                percentage = (non_null / len(df)) * 100
                logger.info(f"  - {col}: {non_null} valores no nulos ({percentage:.2f}%)")
        
        # Verificar si ambos campos están disponibles
        if all(col in df.columns for col in user_cols):
            both_available = (df['NOMBRE DE USUARIO'].notna() & df['USUARIO'].notna()).sum()
            percentage = (both_available / len(df)) * 100
            logger.info(f"  - Registros con ambos campos disponibles: {both_available} ({percentage:.2f}%)")
            
            only_username = (df['NOMBRE DE USUARIO'].notna() & df['USUARIO'].isna()).sum()
            only_user = (df['NOMBRE DE USUARIO'].isna() & df['USUARIO'].notna()).sum()
            
            logger.info(f"  - Registros solo con NOMBRE DE USUARIO: {only_username} ({(only_username/len(df))*100:.2f}%)")
            logger.info(f"  - Registros solo con USUARIO: {only_user} ({(only_user/len(df))*100:.2f}%)")
        
        logger.success("Exploración del CSV completada")
        
        return df
    
    except Exception as e:
        logger.error(f"Error al explorar el CSV: {e}")
        return None

def main():
    """Función principal."""
    logger.info("Iniciando exploración del CSV")
    df = explore_csv()
    
    if df is not None:
        logger.success("Exploración completada con éxito")
    else:
        logger.error("Error en la exploración del CSV")
        sys.exit(1)

if __name__ == "__main__":
    main()
