import logging
import sys
import os

# Intentar importar colorama
try:
    from colorama import init, Fore, Style
    colorama_available = True
    init(autoreset=True)
except ImportError:
    colorama_available = False
    # Se definen valores vacíos para evitar errores si colorama no está
    class Fore:
        CYAN = GREEN = YELLOW = RED = MAGENTA = WHITE = ""
    class Style:
        RESET_ALL = ""

# Crear el directorio y archivo si no existen
log_file_path = "data/app.log"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
if not os.path.exists(log_file_path):
    with open(log_file_path, "w") as f:
        f.write("")  # Crea el archivo vacío

# Crear un logger con un nombre único
logger = logging.getLogger("mi_logger_personalizado")
logger.setLevel(logging.DEBUG)

# Definir colores para cada nivel (si colorama está disponible)
class CustomFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA,
    }

    RESET = Style.RESET_ALL
    FORMAT = "%(asctime)s|%(levelname)s|%(message)s"
    DATEFMT = "%d-%m %H:%M:%S"

    def __init__(self):
        super().__init__(self.FORMAT, datefmt=self.DATEFMT)

    def format(self, record):
        if colorama_available:
            log_color = self.COLORS.get(record.levelname, self.RESET)
            log_message = super().format(record)
            return f"{log_color}{log_message}{self.RESET}"
        else:
            return super().format(record)

# Configurar archivo de logs
file_formatter = logging.Formatter(
    "%(asctime)s|%(levelname)s|[%(filename)s:%(lineno)s|%(funcName)s()]:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = logging.FileHandler(log_file_path, mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)

# Configurar salida a consola
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(CustomFormatter())

# Evita duplicar handlers
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Mensaje de prueba
logger.info("Logger inicializado correctamente.")
