import yaml

def load_config(path="core/config.yaml"):
    "Cargar la configuración desde el archivo YAML"
    with open(path, "r") as f:
        return yaml.safe_load(f)
