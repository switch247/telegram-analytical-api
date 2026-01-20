import logging.config
import yaml
def get_logger(name: str):
    with open("config/logging.yaml") as f:
        logging.config.dictConfig(yaml.safe_load(f))
    return logging.getLogger(name)