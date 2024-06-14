import yaml
import os

def get_config():
    with open('Module_RAG/config_app/config.yaml', encoding='utf-8') as cfgFile:
        config_app = yaml.safe_load(cfgFile)
        cfgFile.close()
    return config_app
