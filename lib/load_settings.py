import math
from typing import Dict, Any

import yaml


def load_settings(path: str) -> Dict[str, Any]:
    """
    Загрузка параметров пайплайна

    :param path: путь до файла с кофигом yaml
    :return: dict
    """
    with open(path, "r") as stream:
        settings = yaml.safe_load(stream)
        settings["MJD_HOUR"] = eval(settings["MJD_HOUR"])
        settings["MJD_TIME_INTERVAL"] = settings["MJD_TIME_INTERVAL"] * settings["MJD_HOUR"]
        settings["extR1"] = settings["r1"]
        settings["PHI0"] = math.radians(settings["PHI0"])
        settings["PHI1"] = math.radians(settings["PHI1"])
        settings["ANOMALY_THRESHOLD_L2"] = settings["LC"] - settings["ANOMALY_THRESHOLD_L2"]
        return settings


def load_settings_cyclone(path: str) -> Dict[str, Any]:
    """
    Загрузка параметров выбранного циклона

    :param path: путь до файла с кофигом yaml
    :return: dict
    """
    with open(path, "r") as stream:
        settings = yaml.safe_load(stream)
        settings["files"] = [settings["TC_NAME"] + "_F"]
        settings["eye_coords"] = settings[settings["TC_NAME"] + "_EYE"]
        settings["out_pressure"] = settings[settings["TC_NAME"] + "_Pinf"]
        settings["eye_lat"] = float(settings["eye_coords"][settings["INDEX"]][0])  # Широта центра ТЦ
        settings["eye_lon"] = float(settings["eye_coords"][settings["INDEX"]][1])  # Долгота центра ТЦ
        return settings
