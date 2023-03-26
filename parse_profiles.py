import os

from lib.load_settings import load_settings, load_settings_cyclone
from lib.math_utils import calc_pressure, plot_pressure
from lib.profile_utils import calc_average_profile, get_sector_profiles
from lib.tc import parse_tc_from_best_track_jma, create_composite_profile

PROFILES_PATH = os.path.join("..", "ProfilesCatalog")
JP_TRACKS_PATH = os.path.join(PROFILES_PATH, "jpTracks", "bst_all.txt")

SETTINGS = load_settings("settings.yaml")
SETTINGS_CYCLONE = load_settings_cyclone("settings_cyclone.yaml")

if __name__ == "__main__":
    for tc_subdir in next(os.walk(PROFILES_PATH))[1]:
        print(tc_subdir)
        if tc_subdir.find("Track") == 0:  # Track_BANYAN_2
            current_tc_name = tc_subdir.split("_")[1]  # BANYAN_2
            try:
                first_folder_name = next(os.walk(os.path.join(PROFILES_PATH, tc_subdir)))[1][
                    0
                ]  # 2012-04-01_0543_NOAA19
                # if first_folder_name and SETTINGS_CYCLONE["TC_NAME"] in current_tc_name: ?
                if first_folder_name:
                    yyyy = first_folder_name.split("-")[0]  # 2012
                    tc_tracks = parse_tc_from_best_track_jma(current_tc_name, yyyy, JP_TRACKS_PATH)

                    # Для каждой координаты из трека ТЦ пытаемся сконструировать композицию из профилей
                    if tc_tracks:
                        for idx, curr_tc in enumerate(tc_tracks):
                            if len(tc_tracks) >= idx + 1 and int(curr_tc.pressure) <= SETTINGS["MAX_PRESSURE_VAL"]:
                                # считаем 'общий' профиль циклона по координатам
                                curr_composite_profiles, lat_list, lon_list = create_composite_profile(
                                    profiles_path=PROFILES_PATH,
                                    tc_sub_dir=tc_subdir,
                                    tc_tracks=tc_tracks,
                                    index=idx,
                                    number_of_channels=SETTINGS["NUMBER_OF_CHANNELS"],
                                    lc=SETTINGS["LC"],
                                    p_num=SETTINGS["pNum"],
                                    delta=SETTINGS["delta"],
                                    ext_r1=SETTINGS["extR1"],
                                    ext_r2=SETTINGS["extR2"],
                                    sat_ignore_list=SETTINGS["SAT_IGNORE_LIST"],
                                    dt=SETTINGS["MJD_TIME_INTERVAL"],
                                )

                                # получаем внешний профиль циклона
                                outer_profile, outer_profiles_num = calc_average_profile(
                                    profiles_array=curr_composite_profiles,
                                    lat_list=lat_list,
                                    lon_list=lon_list,
                                    c_lat=curr_tc.lat,
                                    c_lon=curr_tc.lon,
                                    phi0=SETTINGS["PHI0"],
                                    phi1=SETTINGS["PHI1"],
                                    ext_r1=SETTINGS["extR1"],
                                    ext_r2=SETTINGS["extR2"],
                                    relative_flg=SETTINGS["relative_flg"],
                                )

                                if outer_profiles_num:
                                    sector_profiles, sector_profiles_num = get_sector_profiles(
                                        outer_profile=outer_profile,
                                        profiles_array=curr_composite_profiles,
                                        lat_list=lat_list,
                                        lon_list=lon_list,
                                        tc_lat=curr_tc.lat,
                                        tc_lon=curr_tc.lon,
                                        phi0=SETTINGS["PHI0"],
                                        phi1=SETTINGS["PHI1"],
                                        ext_r1=SETTINGS["extR1"],
                                        lc=SETTINGS["LC"],
                                        anomaly_threshold_l1=SETTINGS["ANOMALY_THRESHOLD_L1"],
                                        anomaly_threshold_l2=SETTINGS["ANOMALY_THRESHOLD_L2"],
                                        anomaly_acceptable_val=SETTINGS["ANOMALY_ACCEPTABLE_VAL"],
                                        relative_flg=SETTINGS["relative_flg"],
                                    )

                                    if sector_profiles_num:
                                        print("Number of sector profiles: %s" % sector_profiles_num)

                                        items = calc_pressure(
                                            sector_profiles,
                                            outer_profile,
                                            curr_tc.lat,
                                            curr_tc.lon,
                                            SETTINGS["relative_flg"],
                                        ).items()
                                        plot_pressure(
                                            items, SETTINGS["r0"], SETTINGS["r1"], SETTINGS["r_stp"], str(curr_tc), ""
                                        )
                                    else:
                                        print("No sector profiles found for TC '" + str(curr_tc) + "'")
                                    break
                                else:
                                    print("No outer profiles found for TC '" + str(curr_tc) + "'")
            except IndexError:
                print(f"broken folder: {tc_subdir}")
