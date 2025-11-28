"""
Configuración de logging con loguru para el proyecto de neurofísica computacional.

Crea logs separados por experimento con timestamp, guardados en results/logs/
"""

from loguru import logger
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    experiment_name: str = "general",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_to_file: bool = True,
    results_dir: Path = None
):
    """
    Configura el logger para el proyecto.
    
    Parámetros
    ----------
    experiment_name : str
        Nombre del experimento (usado en el nombre del archivo)
    console_level : str
        Nivel mínimo para output en consola (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
    file_level : str
        Nivel mínimo para guardar en archivo
    log_to_file : bool
        Si True, guarda logs en archivo
    results_dir : Path, opcional
        Directorio raíz de results/ (si None, lo busca automáticamente)
    
    Retorna
    -------
    logger
        Logger configurado de loguru
    """
    
    # Remover handlers por defecto
    logger.remove()
    
    # Formato personalizado
    console_format = (
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    file_format = (
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )
    
    # Handler para consola (coloreado)
    logger.add(
        sys.stderr,
        format=console_format,
        level=console_level,
        colorize=True
    )
    
    # Handler para archivo (si se solicita)
    if log_to_file:
        # Buscar directorio results/logs
        if results_dir is None:
            # Intentar encontrar results/ desde src/
            current = Path(__file__).resolve()
            for parent in current.parents:
                results_candidate = parent / "results"
                if results_candidate.exists():
                    results_dir = results_candidate
                    break
            
            if results_dir is None:
                # Crear en directorio actual
                results_dir = Path.cwd() / "results"
        
        logs_dir = results_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = logs_dir / f"{experiment_name}_{timestamp}.log"
        
        logger.add(
            log_filename,
            format=file_format,
            level=file_level,
            rotation="100 MB",  # Rotar si supera 100MB
            retention="30 days",  # Mantener logs de último mes
            compression="zip"  # Comprimir logs antiguos
        )
        
        logger.info(f"Logs guardándose en: {log_filename}")
    
    return logger


def get_logger():
    """
    Retorna el logger global de loguru.
    Usar después de llamar setup_logger().
    """
    return logger


# Ejemplo de uso en notebooks/scripts:
if __name__ == "__main__":
    # Setup
    log = setup_logger(
        experiment_name="test_delays",
        console_level="DEBUG",
        log_to_file=True
    )
    
    # Probar niveles
    log.trace("Mensaje TRACE - máximo detalle")
    log.debug("Mensaje DEBUG - información de desarrollo")
    log.info("Mensaje INFO - flujo normal del programa")
    log.success("Mensaje SUCCESS - operación completada exitosamente")
    log.warning("Mensaje WARNING - advertencia")
    log.error("Mensaje ERROR - error recuperable")
    log.critical("Mensaje CRITICAL - error crítico")
    
    # Con contexto adicional
    log.info("Simulación iniciada", extra={"n_neurons": 100, "duration": 1000})