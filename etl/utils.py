"""
Utilidades para el procesamiento de datos en el pipeline ETL.
"""
import re
import unicodedata
import pandas as pd
from typing import List, Dict, Any, Union
from loguru import logger
import spacy
from langdetect import detect, LangDetectException

# Cargar modelo de spaCy para español
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    logger.warning("Modelo de spaCy no encontrado. Ejecuta: python -m spacy download es_core_news_sm")
    nlp = None

def detect_language(text: str) -> str:
    """
    Detecta el idioma de un texto.
    
    Args:
        text (str): Texto a analizar
        
    Returns:
        str: Código del idioma detectado (es, en, etc.)
    """
    if not text or not isinstance(text, str):
        return "unknown"
    
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def clean_text(text: str) -> str:
    """
    Limpia un texto eliminando URLs, menciones, hashtags y normalizando espacios.
    
    Args:
        text (str): Texto a limpiar
        
    Returns:
        str: Texto limpio
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Eliminar URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Eliminar menciones
    text = re.sub(r'@\w+', '', text)
    
    # Eliminar hashtags (opcional, a veces son informativos)
    # text = re.sub(r'#\w+', '', text)
    
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text)
    
    # Normalizar acentos y caracteres especiales
    text = unicodedata.normalize('NFKD', text)
    
    return text.strip()

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extrae entidades nombradas de un texto usando spaCy.
    
    Args:
        text (str): Texto a analizar
        
    Returns:
        Dict[str, List[str]]: Diccionario con entidades por tipo
    """
    if not nlp or not text or not isinstance(text, str):
        return {}
    
    doc = nlp(text)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    
    return entities

def calculate_engagement(row: pd.Series) -> float:
    """
    Calcula el engagement de una publicación basado en interacciones.
    Adaptado para las columnas específicas del dataset de Junta de Andalucía.
    
    Args:
        row (pd.Series): Fila del DataFrame con datos de la publicación
        
    Returns:
        float: Valor de engagement
    """
    engagement = 0
    
    # Si ya existe una columna de engagement, usarla como base
    if 'engagement_original' in row and pd.notna(row['engagement_original']):
        return float(row['engagement_original'])
    
    # Calcular basado en interacciones disponibles
    if 'likes' in row and pd.notna(row['likes']):
        engagement += float(row['likes'])
    
    if 'comments' in row and pd.notna(row['comments']):
        engagement += float(row['comments']) * 2  # Los comentarios tienen más peso
    
    if 'shares' in row and pd.notna(row['shares']):
        engagement += float(row['shares']) * 3  # Los compartidos tienen aún más peso
    
    # Otras métricas específicas del dataset
    if 'REACCIONES' in row and pd.notna(row['REACCIONES']):
        engagement += float(row['REACCIONES'])
    
    if 'INTERACCIONES' in row and pd.notna(row['INTERACCIONES']):
        engagement += float(row['INTERACCIONES'])
    
    # Normalizar por seguidores si está disponible
    if 'followers' in row and pd.notna(row['followers']) and float(row['followers']) > 0:
        engagement = (engagement / float(row['followers'])) * 100
    
    return engagement

def normalize_source(source: str) -> str:
    """
    Normaliza el nombre de la fuente para categorización.
    Adaptado para las fuentes específicas del dataset de Junta de Andalucía.
    
    Args:
        source (str): Nombre de la fuente original
        
    Returns:
        str: Nombre de la fuente normalizado
    """
    if not source or not isinstance(source, str):
        return "unknown"
    
    source = source.lower()
    
    if any(x in source for x in ['twitter', 'tweet', 'x.com', 'x com']):
        return 'twitter'
    elif any(x in source for x in ['facebook', 'fb']):
        return 'facebook'
    elif any(x in source for x in ['instagram', 'ig']):
        return 'instagram'
    elif any(x in source for x in ['tiktok', 'tk']):
        return 'tiktok'
    elif any(x in source for x in ['youtube', 'yt']):
        return 'youtube'
    elif any(x in source for x in ['reddit']):
        return 'reddit'
    elif any(x in source for x in ['blog']):
        return 'blog'
    elif any(x in source for x in ['foro']):
        return 'foro'
    elif any(x in source for x in ['news', 'noticias', 'periódico', 'diario']):
        return 'noticias'
    else:
        return 'web'
