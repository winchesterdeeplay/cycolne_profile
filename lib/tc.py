import os
from typing import List, Optional, Tuple

import numpy as np
from astropy.time import Time

from lib.profile_utils import construct_relative_area, profile_is_bad, read_profiles


class TC:
    def __init__(self, name: str, yyyy: str, mm: str, dd: str, hh: str, lat, lon, pressure):
        self.name = name
        self.yyyy = yyyy
        self.mm = mm
        self.dd = dd
        self.hh = hh
        self.lat = lat
        self.lon = lon
        self.pressure = pressure

    def __str__(self):
        return "%s %s-%s-%sT%s:00 [%s; %s] %s GPa (JMA)" % (
            self.name,
            self.yyyy,
            self.mm,
            self.dd,
            self.hh,
            self.lat,
            self.lon,
            self.pressure,
        )

    def to_time(self):
        time = self.yyyy + "-" + self.mm + "-" + self.dd + "T" + self.hh + ":00:00"
        return Time(time, format="isot", scale="utc")


def parse_tc_from_best_track_jma(tc_name: str, yyyy: str, jp_tracks_path: str) -> List[TC]:
    # ToDo мы тут каждый раз заново читаем файл - лучше где-то хранить
    """
    Получаем трек текущего ТЦ из файла best_track JMA

    :param tc_name: имя циклона
    :param yyyy: год циклона
    :param jp_tracks_path: путь до jp_tracks
    :return:
    """
    curr_tc_local = []

    with open(jp_tracks_path) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            line_split = line.split()
            if line_split[0] == "66666" and len(line_split) > 7:
                bst_tc_name = line_split[7]
                if bst_tc_name and bst_tc_name == tc_name:
                    ln = fp.readline()
                    cnt += 1
                    if ln:
                        yy = ln[0:2]
                    if 0 < int(yy) <= 50:
                        yy = "20" + yy
                        if yy == yyyy:
                            str_spl2 = ln.split()
                            while ln and str_spl2[0] != "66666":
                                mm = ln[2:4]
                                dd = ln[4:6]
                                hh = ln[6:8]
                                lat = float(str_spl2[3])
                                lat = lat / 10
                                lon = float(str_spl2[4])
                                lon = lon / 10
                                pressure = str_spl2[5]
                                tc_local = TC(tc_name, yy, mm, dd, hh, lat, lon, pressure)
                                curr_tc_local.append(tc_local)

                                ln = fp.readline()
                                cnt += 1
                                str_spl2 = ln.split()
                            return curr_tc_local

            line = fp.readline()
            cnt += 1


def get_interpolated_tc_location(tc_tracks: List[TC], index: int, profile_mjd: float) -> Optional[Tuple[float, float]]:
    """
    Выборка всех профилей под конкретное положение ТЦ и их композиция путём заполнения пустот и осреднения
    уже имеющихся данных вертикальных профилей.
    :param tc_tracks:
    :param index:
    :param profile_mjd:
    :return:
    """
    curr_tc_mjd = tc_tracks[index].to_time().mjd
    tc_prof_mjd_diff = tc_tracks[index].to_time().mjd - profile_mjd
    # Дата/время текущей координаты ТЦ больше, чем дата/время файла с профилями
    # Проверка на не выход за границу списка tc_track и, соответственно, существование предыдущей коортинаты ТЦ
    vector = 0
    if tc_prof_mjd_diff > 0 and index - 1 >= 0:
        vector = 1
    # Дата/время текущей координаты ТЦ меньше, чем дата/время файла с профилями
    if tc_prof_mjd_diff < 0 and index + 1 <= len(tc_tracks) - 1:
        vector = -1

    if tc_prof_mjd_diff == 0:
        return tc_tracks[index].lat, tc_tracks[index].lon

    if vector != 0:
        prev_tc_mjd = tc_tracks[index - vector].to_time().mjd
        tc_prev_lat = tc_tracks[index - vector].lat
        tc_prev_lon = tc_tracks[index - vector].lon
        lat_diff = tc_prev_lat - tc_tracks[index].lat
        lon_diff = tc_prev_lon - tc_tracks[index].lon
        # Находим пропорцию в разнице времени между координатами ТЦ и разницей между текущей координатой ТЦ и файла с профилями
        # Далее эту пропорцию будем использовать для интерполяции координат ТЦ на момент даты/времени текущего файла профилей
        tc_mjd_diff = curr_tc_mjd - prev_tc_mjd
        scale = tc_prof_mjd_diff / tc_mjd_diff

        interpolated_lat = float("%.2f" % round(tc_tracks[index].lat + lat_diff * scale, 2))
        interpolated_lon = float("%.2f" % round(tc_tracks[index].lon + lon_diff * scale, 2))

        return interpolated_lat, interpolated_lon

    return None


def create_composite_profile(
    profiles_path: str,
    tc_sub_dir: str,
    tc_tracks: List[TC],
    index: int,
    number_of_channels: int,
    lc: int,
    p_num: int,
    delta: float,
    ext_r1: float,
    ext_r2: float,
    sat_ignore_list: List[str],
    dt: float = 0.25,
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    Создание композитного профиля для конкретного файла с профилями
    :param profiles_path: путь до папки с профилями
    :param tc_sub_dir: поддиректории с профилями
    :param tc_tracks: список с циклонами (TC) JMA
    :param index: индекс текущего циклона в списке tc_tracks
    :param number_of_channels: количество каналов
    :param lc: число стандартных метеорологических уровней давления
    :param p_num: число точек, на которые разбивается интервал шириной +-delta
    :param delta: градусы
    :param ext_r1: внешний радиус циклона 1
    :param ext_r2: внешний радиус циклона 2
    :param sat_ignore_list: список игнорирумых спутников
    :param dt: ?
    :return: композитный профиль, список с координатами широт, список с координатами долгот
    """
    lat_list = []
    lon_list = []

    tc_mjd = tc_tracks[index].to_time().mjd

    selected_profiles = []
    selected_profiles_tc_location = []
    nearest_locations = []
    sat = ""

    # Выборка файлов с профилями, подходящими под указанный интервал времени (+-6 часов = 0.25 MJD);
    for profile_gz in next(os.walk(os.path.join(profiles_path, tc_sub_dir)))[2]:
        profile_time = ""
        if profile_gz.find("profiles.gz") > 0:
            yyyy = profile_gz[0:4]
            mm = profile_gz[5:7]
            dd = profile_gz[8:10]
            hh = profile_gz[11:13]
            mins = profile_gz[13:15]
            sat = profile_gz[16:22]
            profile_time = Time(f"{yyyy}-{mm}-{dd}T{hh}:{mins}:00", format="isot", scale="utc")

        if profile_time != "" and sat not in sat_ignore_list:
            if abs(tc_mjd - profile_time.mjd) <= dt:
                curr_tc_location = get_interpolated_tc_location(tc_tracks, index, profile_time.mjd)

                if curr_tc_location is not None:
                    selected_profiles_tc_location.append(curr_tc_location)
                    selected_profiles.append(os.path.join(profiles_path, tc_sub_dir, profile_gz))

    # Создание отсортированного массива координатной сетки по широте и долготе
    # for current_profile_path in selected_profiles:
    #     lat_list, lon_list = construct_area(current_profile_path, lat_list, lon_list, lat, lon, extR2)
    for current_profile_path, curr_tc_location in zip(selected_profiles, selected_profiles_tc_location):
        lat = curr_tc_location[0]
        lon = curr_tc_location[1]
        lat_list, lon_list = construct_relative_area(current_profile_path, lat_list, lon_list, lat, lon, ext_r2)
        nearest_locations.append([lat, lon])

    # Вычисляем "разрешение" сетки профилей: для этого находим минимум по расстоянию по широте и долготе в lat_list и lon_list

    # Массив всех профилей для данного ТЦ
    profiles_array = np.zeros((len(lat_list), len(lon_list), lc))
    # Заполнение массива общего поля вертикальных атмосферных профилей температуры
    sum_array = np.zeros((len(lat_list), len(lon_list)))
    for current_profile_path, location in zip(selected_profiles, selected_profiles_tc_location):
        read_profiles(
            profiles_array=profiles_array,
            profile_fn=current_profile_path,
            tc_lat=location[0],
            tc_lon=location[1],
            lat_list=lat_list,
            lon_list=lon_list,
            ext_r1=ext_r1,
            ext_r2=ext_r2,
            delta=delta,
            lc=lc,
            p_num=p_num,
            number_of_channels=number_of_channels,
            relative_flg=True,
            sum_array=sum_array,
        )

    for lat_index in range(np.size(profiles_array, 0)):
        for lon_index in range(np.size(profiles_array, 1)):
            if not profile_is_bad(profiles_array[lat_index, lon_index]) and sum_array[lat_index, lon_index] > 0:
                for z in range(np.size(profiles_array, 2)):
                    profiles_array[lat_index, lon_index, z] = float(
                        "%.3f" % round(profiles_array[lat_index, lon_index, z] / sum_array[lat_index, lon_index], 3)
                    )

    return profiles_array, lat_list, lon_list
