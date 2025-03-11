"""
Punto de entrada principal para el sistema RAG de análisis de datos sociales.
Este script permite ejecutar tanto el frontend como las utilidades de ETL.
PRUEBA TEST
"""
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Configurar logger
logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}", level="INFO")
logger.add("logs/app.log", rotation="10 MB", retention="1 week", level="DEBUG")

# Cargar variables de entorno
load_dotenv()

def run_frontend():
    """Ejecuta la aplicación frontend de Streamlit."""
    try:
        logger.info("Iniciando la aplicación frontend...")
        os.system("streamlit run frontend/app.py")
    except Exception as e:
        logger.error(f"Error al iniciar la aplicación frontend: {e}")
        sys.exit(1)

def run_etl_pipeline(step=None):
    """
    Ejecuta el pipeline de ETL.
    
    Args:
        step (str): Paso específico del pipeline a ejecutar ('preprocess', 'vectorstore', 'all').
    """
    try:
        if step == 'preprocess' or step == 'all':
            logger.info("Ejecutando preprocesamiento de datos...")
            os.system("python etl/preprocess_data.py")
        
        if step == 'vectorstore' or step == 'all':
            logger.info("Creando vectorstore...")
            os.system("python etl/create_vectorstore.py")
            
        logger.success("Pipeline ETL completado con éxito")
    except Exception as e:
        logger.error(f"Error en el pipeline ETL: {e}")
        sys.exit(1)

def main():
    """Función principal que maneja los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Sistema RAG para análisis de datos sociales")
    
    # Subparsers para diferentes comandos
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
    # Comando 'run' para ejecutar la aplicación
    run_parser = subparsers.add_parser("run", help="Ejecutar la aplicación frontend")
    
    # Comando 'etl' para ejecutar el pipeline ETL
    etl_parser = subparsers.add_parser("etl", help="Ejecutar el pipeline ETL")
    etl_parser.add_argument(
        "--step", 
        choices=["preprocess", "vectorstore", "all"], 
        default="all",
        help="Paso específico del pipeline ETL a ejecutar"
    )
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Ejecutar el comando correspondiente
    if args.command == "run":
        run_frontend()
    elif args.command == "etl":
        run_etl_pipeline(args.step)
    else:
        # Si no se especifica comando, mostrar ayuda
        parser.print_help()

if __name__ == "__main__":
    # Crear directorios necesarios si no existen
    Path("logs").mkdir(exist_ok=True)
    Path("data/raw").mkdir(exist_ok=True, parents=True)
    Path("data/processed").mkdir(exist_ok=True, parents=True)
    Path("data/vectorstore").mkdir(exist_ok=True, parents=True)
    Path("data/reports").mkdir(exist_ok=True, parents=True)
    
    main()
