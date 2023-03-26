import gzip
import math
import os
from typing import List, TextIO, Tuple

import numpy as np
from global_land_mask import globe

from lib.math_utils import calc_dist, correct_lon, correct_lat, cart2pol, pLevels


class Profile:
    def __init__(self, lat, lon, profile):
        self.lat = lat
        self.lon = lon
        self.profile = profile


def open_profile(profile_fn: str) -> TextIO:
    """
    Прочитать данные профиля
    :param profile_fn: путь до профиля
    :return: данные профиля TextIO
    """
    filename, file_extension = os.path.splitext(profile_fn)

    if file_extension == ".gz":
        fs = gzip.open(profile_fn, "rt")
    else:
        fs = open(profile_fn, "r")
    return fs


def construct_relative_area(
    profile_fn: str, lat_list: List[float], lon_list: List[float], tc_lat: float, tc_lon: float, ext_r2: float
) -> Tuple[List[float], List[float]]:
    """

    :param profile_fn: путь до профиля
    :param lat_list: список широт
    :param lon_list: список долгот
    :param tc_lat: широта центра
    :param tc_lon: долгота центра
    :param ext_r2: внешний радиус
    :return: отсортированный список широт, отсортированный список долгот
    """
    fs = open_profile(profile_fn)

    obs_id_counter = 0

    # Пересчёт координат в смещения относительно найденного профиля с координатами [min_lat; min_lon]:
    line = fs.readline()
    while True:
        if not line:
            break

        if "Obs ID:" in line:
            obs_id_counter += 1
            line = fs.readline()

            if "Latitude:" in line:
                str_spl = line.split()
                curr_lat = float("%.2f" % round(float(str_spl[1]), 2))
                curr_lon = float("%.2f" % round(float(str_spl[3]), 2))

                dist = calc_dist(curr_lat, curr_lon, tc_lat, tc_lon)

                if dist <= ext_r2:
                    for i in range(12):
                        line = fs.readline()
                    if "Obs ID:" not in line:
                        curr_lat_rel = round(curr_lat - tc_lat, 2)
                        curr_lon_rel = round(curr_lon - tc_lon, 2)

                        if curr_lat_rel not in lat_list:
                            lat_list.append(curr_lat_rel)

                        if curr_lon_rel not in lon_list:
                            lon_list.append(curr_lon_rel)

                        line = fs.readline()
                    else:
                        print(line)
        else:
            line = fs.readline()

    fs.close()

    lat_list.sort(key=lambda x: float(x))
    lon_list.sort(key=lambda x: float(x))

    return lat_list, lon_list


def profile_is_bad(profile: List[float]) -> bool:
    """
    Проверка профиля на наличие нулевых значений
    :param profile: список значений профиля
    :return: True/False
    """
    for z in range(len(profile)):
        if (
            profile[z] <= 0.0
            or not isinstance(profile[z], float)
            or math.isnan(profile[z])
            or not isinstance(profile[z], float)
        ):
            return True
    return False


def is_ocean(lat: float, lon: float, stp: float, n: int):
    """
    Проверка наличия океана в заданном радиусе
    :param lat: долгота
    :param lon: широта
    :param stp: шаг
    :param n: количество шагов
    :return: True/False
    """
    lat_np = np.linspace(lat - stp, lat + stp, n)
    lon_np = np.linspace(lon - stp, lon + stp, n)

    lat = correct_lat(lat)
    lon = correct_lon(lon)

    for i, curr_lat in enumerate(lat_np):
        lat_np[i] = correct_lat(lat_np[i])

    for j, curr_lon in enumerate(lon_np):
        lon_np[j] = correct_lon(lon_np[j])

    lon_array = globe.is_land(lat, lon_np)
    lat_array = globe.is_land(lat_np, lon)

    if True in lon_array or True in lat_array:
        return False
    else:
        return True


def read_profiles(
    profiles_array: np.ndarray,
    profile_fn: str,
    tc_lat: float,
    tc_lon: float,
    lat_list: List[float],
    lon_list: List[float],
    delta: float,
    ext_r1: float,
    ext_r2: float,
    lc: int,
    p_num: int,
    number_of_channels=15,
    relative_flg=False,
    sum_array=None,
) -> None:
    """
    Чтение профилей температуры из файла

    :param profiles_array: массив профилей текущего тц
    :param profile_fn: путь к файлу с профилями
    :param tc_lat: ширина центра циклона
    :param tc_lon: долгота центра циклона
    :param lat_list: список широт
    :param lon_list: список долгот
    :param profiles_array: список профилей температуры
    :param delta: градусы
    :param ext_r1: внешний радиус 1
    :param ext_r2: внешний радиус 2
    :param lc: число точек, на которые разбивается интервал шириной +-delta
    :param p_num: число стандартных метеорологических уровней давления
    :param number_of_channels: количество каналов
    :param relative_flg: ?
    :param sum_array: ?
    :return: количество профилей, массив профилей
    """
    # !!! Обратить внимание на этот файл в плане "глаза" от JMA - 2011-07-20_0559_NOAA15.joined_satellite_data.profiles
    # files = '2011-06-24_1625_NOAA19.joined_satellite_data.profiles'

    # 3-х мерный массив профилей температуры [lat, lon, level] = temp (K);
    # profiles_array = np.zeros((len(lat_list), len(lon_list), LC))

    fs = open_profile(profile_fn)

    obs_id_counter = 0
    current_number_of_channels = 0

    curr_lat = -1.0
    curr_lon = -1.0

    _profiles_num_ = 0

    line = fs.readline()
    while True:
        if not line:
            break

        if "Obs ID:" in line:
            obs_id_counter += 1
            line = fs.readline()

            if "Latitude:" in line:
                str_spl = line.split()
                curr_lat = float("%.2f" % round(float(str_spl[1]), 2))
                curr_lon = float("%.2f" % round(float(str_spl[3]), 2))

                for i in range(11):
                    line = fs.readline()

                if "Number of Channels Used =" in line:
                    current_number_of_channels = int(line.split("Number of Channels Used =")[1])

                for i in range(current_number_of_channels + 3):
                    line = fs.readline()

                if "Pressure (Pa)  T (K)" in line:
                    # Фильтрация профилей над сушей
                    if is_ocean(curr_lat, curr_lon, delta, p_num):
                        # if curr_lat >= 0.0 and curr_lon >= 0.0:
                        dist = float("%.4f" % round(calc_dist(curr_lat, curr_lon, float(tc_lat), float(tc_lon)), 4))

                        # Осуществляем фильтрацию текущего профиля по числу задействованных при его расчётах каналов
                        if (dist <= ext_r1 or ext_r1 < dist < ext_r2) and (
                            current_number_of_channels >= number_of_channels
                        ):
                            # print("number_of_channels = ", number_of_channels, "Number of channels used = ", current_number_of_channels)

                            if relative_flg:
                                curr_lat = float("%.2f" % round(curr_lat - tc_lat, 2))
                                curr_lon = float("%.2f" % round(curr_lon - tc_lon, 2))

                            if curr_lat in lat_list and curr_lon in lon_list:
                                curr_profile = np.zeros(lc)

                                for z in range(lc):
                                    line = fs.readline()
                                    curr_temp = float(line.split()[1].strip())
                                    if isinstance(curr_temp, float):
                                        curr_profile[z] = float(curr_temp)
                                    else:
                                        print("Error: current temperature '", curr_temp, "' is NOT a digid!\n")
                                        break

                                if not profile_is_bad(curr_profile):
                                    _profiles_num_ += 1
                                    for z in range(lc):
                                        profiles_array[
                                            int(lat_list.index(curr_lat)), int(lon_list.index(curr_lon)), z
                                        ] += curr_profile[z]

                                    # При необходимости подсчитываем количество профилей по текущим координатам
                                    if sum_array is not None:
                                        sum_array[int(lat_list.index(curr_lat)), int(lon_list.index(curr_lon))] += 1
                else:
                    line = fs.readline()
        else:
            line = fs.readline()

    fs.close()


def calc_average_profile(
    profiles_array: np.ndarray,
    lat_list: List[float],
    lon_list: List[float],
    c_lat: float,
    c_lon: float,
    phi0: float,
    phi1: float,
    ext_r1: float,
    ext_r2: float,
    relative_flg: bool = False,
):
    """
    Вычисление среднего профиля по заданным ограничениям координат

    :param profiles_array: массив профилей
    :param lat_list: список широт
    :param lon_list: список долгот
    :param c_lat: широта центра ТЦ
    :param c_lon: долгота центра ТЦ;
    :param phi0: малый радиус
    :param phi1: большой радиус
    :param ext_r1: внешний радиус 1
    :param ext_r2: внутренний радиус 2
    :param relative_flg:
    :return:

    """

    lat_num = np.size(profiles_array, 0)
    lon_num = np.size(profiles_array, 1)
    prof_count = 0
    average_profile = np.zeros(np.size(pLevels))
    phi_flg = False

    for lat in range(lat_num):
        for lon in range(lon_num):
            curr_lat = lat_list[lat]
            curr_lon = lon_list[lon]

            if relative_flg:
                rho, phi = cart2pol(curr_lat, curr_lon)
            else:
                lat_val = float("%.2f" % round(curr_lat - c_lat, 2))
                lon_val = float("%.2f" % round(curr_lon - c_lon, 2))
                rho, phi = cart2pol(lat_val, lon_val)

            if phi < 0:
                phi = abs(phi) + math.pi

            # currDist = calcDist(curr_lat, curr_lon, cLat, cLon)
            if phi1 < phi0:
                phi1 += 2 * math.pi
                phi_flg = True

            if phi_flg:
                phi += 2 * math.pi
            if ext_r1 <= rho <= ext_r2 and phi0 <= phi <= phi1:
                if not profile_is_bad(profiles_array[lat, lon]):
                    for z in range(np.size(average_profile)):
                        average_profile[z] += profiles_array[lat, lon, z]
                    prof_count += 1

    if prof_count > 0:
        for z in range(np.size(average_profile, 0)):
            average_profile[z] /= prof_count

    return average_profile, prof_count


def is_profile_anomaly(
    profile, outer_profile, anomaly_threshold_l1, anomaly_threshold_l2, anomaly_acceptable_val, lc
) -> bool:
    if anomaly_threshold_l1 < 0:
        anomaly_threshold_l1 = 0
    if anomaly_threshold_l2 > lc - 1:
        anomaly_threshold_l2 = lc - 1

    for i in range(anomaly_threshold_l1, anomaly_threshold_l2, 1):
        if profile[i] - outer_profile[i] < anomaly_acceptable_val:
            return True
    return False


def get_sector_profiles(
    outer_profile,
    profiles_array,
    lat_list,
    lon_list,
    tc_lat,
    tc_lon,
    phi0,
    phi1,
    ext_r1,
    lc,
    anomaly_threshold_l1: float,
    anomaly_threshold_l2: float,
    anomaly_acceptable_val: float,
    relative_flg=False,
):
    profile_counter = 0
    sector_profiles = []

    lat_num = np.size(profiles_array, 0)
    lon_num = np.size(profiles_array, 1)
    phi_flg = False

    for lat_ind in range(lat_num):
        for lon_ind in range(lon_num):
            curr_lat = lat_list[lat_ind]
            curr_lon = lon_list[lon_ind]

            if relative_flg:
                rho, phi = cart2pol(curr_lat, curr_lon)
            else:
                lat_val = float("%.2f" % round(curr_lat - tc_lat, 2))
                lon_val = float("%.2f" % round(curr_lon - tc_lon, 2))
                rho, phi = cart2pol(lat_val, lon_val)

            if phi < 0:
                phi = abs(phi) + math.pi

            if phi1 < phi0:
                phi1 += 2 * math.pi
                phi_flg = True

            if phi_flg:
                phi += 2 * math.pi

            if rho <= ext_r1 and phi0 <= phi <= phi1:
                curr_profile = np.zeros(lc)

                if not profile_is_bad(profiles_array[lat_ind, lon_ind]):
                    for z in range(np.size(pLevels)):
                        curr_profile[z] = profiles_array[lat_ind, lon_ind, z]

                    profile = Profile(curr_lat, curr_lon, curr_profile)

                    # Исключение аномальных профилей
                    if not profile_is_bad(profile.profile):
                        check_exist = False
                        if not is_profile_anomaly(
                            profile.profile,
                            outer_profile,
                            anomaly_threshold_l1,
                            anomaly_threshold_l2,
                            anomaly_acceptable_val,
                            lc,
                        ):
                            if profile_counter == 0:
                                sector_profiles.insert(profile_counter, profile)

                            if profile_counter > 0:
                                # Проверка профилей на уникальность
                                for i in range(np.size(sector_profiles)):
                                    if str(profile.lat) == str(sector_profiles[i].lat) and str(profile.lon) == str(
                                        sector_profiles[i].lon
                                    ):
                                        check_exist = True
                                        break

                            # Добавляем только уникальные профили в массив
                            if not check_exist:
                                sector_profiles.insert(profile_counter, profile)
                                profile_counter += 1

    return np.array(sector_profiles), profile_counter
