"""
Configuración y funciones para el modelo de lenguaje (LLM).
"""
import os
from dotenv import load_dotenv
from loguru import logger
from typing import Dict, Any, List, Optional

from langchain_ollama import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Cargar variables de entorno
load_dotenv()

def get_llm(model_name: Optional[str] = None, temperature: float = 0.1, streaming: bool = False) -> Ollama:
    """
    Inicializa y configura el modelo de lenguaje.
    
    Args:
        model_name (Optional[str]): Nombre del modelo en Ollama
        temperature (float): Temperatura para la generación (0.0-1.0)
        streaming (bool): Si se debe usar streaming para la salida
        
    Returns:
        Ollama: Instancia configurada del LLM
    """
    # Usar modelo de variables de entorno si no se especifica
    if model_name is None:
        model_name = os.getenv("LLM_MODEL", "granite3.2")
    
    logger.info(f"Inicializando LLM: {model_name} (temperatura: {temperature})")
    
    # Configurar callbacks para streaming si es necesario
    callback_manager = None
    if streaming:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Inicializar el modelo
    llm = Ollama(
        model=model_name,
        temperature=temperature,
        callback_manager=callback_manager if streaming else None,
        verbose=True
    )
    
    return llm

def test_llm(llm: Ollama, prompt: str = "Hola, ¿cómo estás?") -> str:
    """
    Prueba el modelo de lenguaje con un prompt simple.
    
    Args:
        llm (Ollama): Instancia del LLM
        prompt (str): Prompt de prueba
        
    Returns:
        str: Respuesta del modelo
    """
    logger.info(f"Probando LLM con prompt: '{prompt}'")
    
    try:
        response = llm.invoke(prompt)
        logger.success("LLM respondió correctamente")
        return response
    except Exception as e:
        logger.error(f"Error al probar el LLM: {e}")
        raise

if __name__ == "__main__":
    # Código de prueba
    llm = get_llm()
    response = test_llm(llm)
    print(f"Respuesta del LLM: {response}")
