import math
import os
import statistics as stat
from collections import OrderedDict
from pathlib import Path
from typing import Tuple, Dict

import matplotlib.pylab as plt
import numpy as np

from lib.load_settings import load_settings, load_settings_cyclone


class Pressure:
    def __init__(self, sat_name, pressure):
        self.satName = sat_name
        self.pressure = pressure


SETTINGS = load_settings("settings.yaml")
SETTINGS_CYCLONE = load_settings_cyclone("settings_cyclone.yaml")

# Массив уровней давления
pLevels = np.array(
    [
        1013.25,  # 0
        1005.43,  # 1
        985.88,  # 2
        957.44,  # 3
        922.46,  # 4
        882.8,  # 5
        839.95,  # 6
        795.09,  # 7
        749.12,  # 8
        702.73,  # 9
        656.43,  # 10
        610.6,  # 11
        565.54,  # 12
        521.46,  # 13
        478.54,  # 14
        436.95,  # 15
        396.81,  # 16
        358.28,  # 17
        321.5,  # 18
        286.6,  # 19
        253.71,  # 20
        222.94,  # 21
        194.36,  # 22
        167.95,  # 23
        143.84,  # 24
        122.04,  # 25
        102.05,  # 26
        85.18,  # 27
        69.97,  # 28
        56.73,  # 29
        45.29,  # 30
        35.51,  # 31
        27.26,  # 32
        20.4,  # 33
        14.81,  # 34
        10.37,  # 35
        6.95,  # 36
        4.41,  # 37
        2.61,  # 38
        1.42,  # 39
        0.69,  # 40
        0.29,  # 41
        0.1,  # 42
    ]
)


def calc_dist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Приближенное вычисление радиуса земли в киллометры
    :param lat1: долгота 1
    :param lon1: ширина 1
    :param lat2: долгота 2
    :param lon2: ширина 2
    :return: дистанция км
    """
    # approximate radius of earth in km
    R = 6373.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance / 111.0


def cart2pol(x, y) -> Tuple[float, float]:
    """
    Перевод координат из декартовой системы в полярную
    :param x: координата x
    :param y: координата y
    :return: радиус, угол
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi) -> Tuple[float, float]:
    """
    Перевод координат из полярной системы в декартовую
    :param rho: радиус
    :param phi: угол
    :return: координаты x, y
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def correct_lon(lon: float) -> float:
    """
    Вычисление долготы
    :param lon: долгота
    :return: долгота
    """
    if lon > 180.0:
        lon = -(180.0 - lon % 180.0)
    if lon < -180.0:
        lon = 180.0 - lon % 180.0

    return float("%.2f" % round(lon, 2))


def correct_lat(lat: float) -> float:
    """
    Вычисление широты
    :param lat: широта
    :return: широта
    """
    if lat > 90.0:
        lat = -(90.0 - lat % 90.0)
    if lat < -90.0:
        lat = 90.0 - lat % 00.0

    return float("%.2f" % round(lat, 2))


def calc_sp_by_height(
    profile_inside_tc: np.ndarray,
    profile_outside_tc: np.ndarray,
    cG: float = 9.80665,
    cM: float = 0.0289644,
    cR: float = 8.31447,
) -> float:
    """
    Пошаговое вычисление давления на первом уровне ТЦ

    :param profile_inside_tc: внутренний профиль
    :param profile_outside_tc: внешний профиль
    :param cG: Ускорение свободного падения
    :param cM: Молярная масса воздуха
    :param cR: Универсальная газовая постоянная
    :return: Давление на первом уровне внутри ТЦ
    """

    lc = np.size(pLevels)
    start_l = 5  # уровень, с которого начинается расчёт по профилям
    outer_h = np.zeros(lc)  # массив высот на внешней стороне ТЦ;
    int_h = np.zeros(lc)  # массив высот, внутри ТЦ;
    aver_t = 0.0

    # Вычисляем высоты уровней среднего профиля вокруг ТЦ
    # Вычисляем среднюю температуру для первых
    for i in range(start_l):
        aver_t += profile_outside_tc[i]
    aver_t /= start_l

    # Выяснить, почему в конце формулы ноль!!!???
    g_ = cG * (
        1.0
        - 0.002637 * math.cos(2 * SETTINGS_CYCLONE["eye_lat"])
        + 0.0000059 * math.cos(2 * SETTINGS_CYCLONE["eye_lat"]) ** 2 * (1 - 0.000000309 * 0)
    )
    outer_h[start_l - 1] = 0 - math.log(pLevels[start_l - 1] / pLevels[0]) * ((cR * aver_t) / (cM * g_))

    last_h_index = 0

    for i in range(start_l - 1, lc - 1):
        g_ = cG * (
            1.0
            - 0.002637 * math.cos(2 * SETTINGS_CYCLONE["eye_lat"])
            + 0.0000059 * math.cos(2 * SETTINGS_CYCLONE["eye_lat"]) ** 2 * (1 - 0.000000309 * outer_h[i])
        )
        outer_h[i + 1] = outer_h[i] - math.log(pLevels[i + 1] / pLevels[i]) * (
            cR * ((profile_outside_tc[i + 1] + profile_outside_tc[i]) / 2) / (cM * g_)
        )
        last_h_index = i + 1

    # Вычисляем высоты уровней профиля внутри ТЦ
    g_ = cG * (
        1.0
        - 0.002637 * math.cos(2 * SETTINGS_CYCLONE["eye_lat"])
        + 0.0000059 * math.cos(2 * SETTINGS_CYCLONE["eye_lat"]) ** 2 * (1 - 0.000000309 * outer_h[last_h_index])
    )
    int_h[last_h_index - 1] = outer_h[last_h_index] - (
        math.log(pLevels[last_h_index - 1] / pLevels[last_h_index])
        * (cR * ((profile_inside_tc[last_h_index - 1] + profile_inside_tc[last_h_index]) / 2) / (cM * g_))
    )

    for i in range(last_h_index - 2, -1, -1):
        g_ = cG * (
            1.0
            - 0.002637 * math.cos(2 * SETTINGS_CYCLONE["eye_lat"])
            + 0.0000059 * math.cos(2 * SETTINGS_CYCLONE["eye_lat"]) ** 2 * (1 - 0.000000309 * int_h[i + 1])
        )
        int_h[i] = int_h[i + 1] - (
            math.log(pLevels[i] / pLevels[i + 1])
            * (cR * (profile_inside_tc[i] + profile_inside_tc[i + 1]) / 2)
            / (cM * g_)
        )
    aver_t = 0.0
    for i in range(start_l):
        aver_t += profile_inside_tc[i]
    aver_t /= start_l

    # Вычисляем давление на первом уровне профиля внутри ТЦ
    surface_pressure = pLevels[start_l - 1] / (math.exp((-cG * cM * int_h[start_l - 1]) / (cR * aver_t)))

    return surface_pressure


def calc_pressure(
    input_profiles: np.ndarray,
    outer_profile: np.ndarray,
    center_lat: float,
    center_lon: float,
    relative_flg: bool = False,
) -> Dict[float, float]:
    """
    Вычисляем давление в центре циклона

    :param profile_inside_tc: внутренний профиль
    :param profile_outside_tc: внешний профиль
    :param center_lat: координата центра циклона (широта)
    :param center_lon: координата центра циклона (долгота)
    :param relative_flg:
    :return: словарь mu/pressure
    """
    pressure = []
    _r_ = []
    press_to_plot = {}
    num_of_press_in_r = {}

    if relative_flg:
        center_lat = 0.0
        center_lon = 0.0

    for i in range(0, np.size(input_profiles) - 1):
        # значение давления по профилям
        surface_pressure = calc_sp_by_height(input_profiles[i].profile, outer_profile)
        # print(surface_pressure)

        # корректировка давления на разницу между стандартным и реальными
        surface_pressure -= SETTINGS["SLP"] - float(SETTINGS_CYCLONE["out_pressure"][SETTINGS_CYCLONE["INDEX"]])

        lat1 = input_profiles[i].lat
        lon1 = input_profiles[i].lon

        if relative_flg:
            delta_lat = float("%.2f" % round(center_lat + lat1, 2))
            delta_lon = float("%.2f" % round(center_lon + lon1, 2))
            r = calc_dist(delta_lat, delta_lon, float(center_lat), float(center_lon))
        else:
            r = calc_dist(center_lat, center_lon, lat1, lon1)

        pressure.append(surface_pressure)
        if r in press_to_plot:
            press_to_plot[r] += surface_pressure
            num_of_press_in_r[r] += 1
        else:
            press_to_plot[r] = surface_pressure
            num_of_press_in_r[r] = 1

    for key in press_to_plot:
        press_to_plot[key] /= num_of_press_in_r[key]

    return press_to_plot


def plot_pressure(
        items: Dict[float, float],
        r0: float,
        r1: float,
        r_step: int,
        tc_name: str,
        log_str: int = SETTINGS_CYCLONE["INDEX"]
) -> None:
    """
    Отрисовка Pressure и расстояния

    :param items: словарь mu/pressure
    :param r0: радиус 1
    :param r1: радиус 2
    :param r_step: шаг радиуса
    :param tc_name: название циклона
    :param log_str:
    """

    def median(sp_dict, _r_stp_):
        max_r = max(sp_dict.keys())
        _iter_ = 1
        result_array = {}

        while _r_stp_ * _iter_ <= max_r + _r_stp_:
            curr_array = []
            for key in sp_dict:
                if _r_stp_ * (_iter_ - 1) <= key < _r_stp_ * _iter_:
                    curr_array.append(sp_dict[key])
            if np.size(curr_array) > 0:
                result_array[_r_stp_ * _iter_] = stat.median(curr_array)
            _iter_ += 1

        return result_array

    def average(sp_dict, _r_stp_):
        max_r = max(sp_dict.keys())
        _iter_ = 1
        result_array = {}

        while _r_stp_ * _iter_ <= max_r + _r_stp_:
            curr_array = []
            for key in sp_dict:
                if _r_stp_ * (_iter_ - 1) <= key < _r_stp_ * _iter_:
                    curr_array.append(sp_dict[key])
            if np.size(curr_array) > 0:
                result_array[_r_stp_ * _iter_] = np.average(curr_array)
            _iter_ += 1

        return result_array

    press_to_plot = OrderedDict(sorted(items, key=lambda x: x[0]))

    press_to_plot_median = median(press_to_plot, r_step)
    press_to_plot_average = average(press_to_plot, r_step)

    _str_ = ""
    for key, value in press_to_plot_median.items():
        # print('[' + "%05.2f" % key + ']', ': ', ("%06.2f" % value) + ' GPa')
        _str_ += "\t[" + ("%05.2f" % key) + "]" + ": " + "%06.2f" % value + " GPa; "
    print("Median:\n" + _str_)

    _str_ = ""
    for key, value in press_to_plot_average.items():
        # print('[' + "%05.2f" % key + ']', ': ', ("%06.2f" % value) + ' GPa')
        _str_ += "\t[" + ("%05.2f" % key) + "]" + ": " + "%06.2f" % value + " GPa; "
    print("Average:\n" + _str_ + "\n")

    # if press_to_plot_average:
    #     print('\n')

    # fig = plt.figure(figsize=(7, 5))  # размер области построения графиков
    fig = plt.figure()  # размер области построения графиков
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_xlabel("mu = r/r0")
    ax.set_ylabel("Pressure (GPa)")
    _str_ = str(tc_name) + " " + str(log_str)
    ax.set_title(_str_)
    ax.set_xlim(r0, r1)
    max_pressure = round(max(press_to_plot.values()) + 5.0)
    min_pressure = round(min(press_to_plot.values()) - 5.0)
    ax.set_ylim(min_pressure, max_pressure)
    ax.plot(list(press_to_plot_median.keys()), list(press_to_plot_median.values()), color="green", linewidth=2)
    ax.plot(list(press_to_plot_average.keys()), list(press_to_plot_average.values()), color="blue", linewidth=2)
    ax.scatter(list(press_to_plot.keys()), list(press_to_plot.values()), color="red")
    path = os.path.join(
        Path(os.path.abspath(__file__)).parent.parent, "plots", str(tc_name) + "_" + str(log_str) + str(".png")
    )
    plt.savefig(path)
    plt.close()
