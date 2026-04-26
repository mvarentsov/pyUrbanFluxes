

def config_vars (lang):
    if lang == 'ru':
        vars2draw_labels = {'T_air'  : 'Температура воздуха $T, °C$',
                            'Wind_vel'  : 'Скорость ветра $V, м/с$',
                            #'w'  : 'Верт. скорость ветра, м/с',
                            'Qtau': 'Поток импульса $\\tau, кг\\cdot м^{-2}\\cdot c^{-1}$',
                            'Qh'  : 'Поток явного тепла $H, Вт\\cdotм^{-2}$',
                            'Qle' : 'Поток скрытого тепла $LE, Вт\\cdotм^{-2}$',
                            'Cd'  : 'Коэффициент обмена $C_d$'}
    else:
        vars2draw_labels = {'T_air'  : 'Air temperature $T, °C$',
                            'Wind_vel'  : 'Wind speed $V, m/s$',
                            'Qtau': 'Momentum flux $\\tau, kg·m^{-2}·s^{-1}$',
                            'Qh'  : 'Sensible heat flux $H, W·m^{-2}$',
                            'Qle' : 'Latent heat flux $LE, W·m^{-2}$',
                            'Cd'  : 'Exchange coefficient $C_d$'}
    return vars2draw_labels

def config_city (city_name, lang):

    site_names2draw = {}
    site_colors = {}
    site_styles = {}
    
    if city_name == 'Moscow':
        site_names2draw = {s:s for s in ['MSU_19m', 'MSU_11m', 'ZNS_56m']}
        site_colors = {'MSU_19m':'darkorange', 'MSU_11m':'orange', 'ZNS_56m':'blue'}

        site_styles = {'MSU_11m': '--'}

    elif city_name == 'Beijing':
        if lang == 'ru':
            site_names2draw = {'Xianghe_32m':'Фон (32 м)',
                            'Xianghe_80m':'Фон (80 м)',
                            'Beijing_47m': 'Город (47 м)', 
                            'Beijing_80m': 'Город (80 м)'}
        else:
            site_names2draw = {'Xianghe_32m':'Xianghe (32 m)',
                            'Xianghe_80m':'Xianghe (80 m)',
                            'Beijing_47m': 'Beijing (47 m)', 
                            'Beijing_80m': 'Beijing (80 m)'}

        site_colors = {'Xianghe_32m':'blue',
                    'Xianghe_80m':'lightblue',
                    'Beijing_47m': 'red',
                    'Beijing_80m': 'orange'}

    elif city_name == 'Tomsk':

        if lang == 'ru':
            site_names2draw = {'BEK_30m':'Фон (БЭК)',
                            'Voronino_10m': 'Фон (Воронино)',
                            'IMCES_40m': 'Город (ИМКЭС, 40 м)', 
                            'IMCES_30m': 'Город (ИМКЭС, 30 м)',
                            'IMCESmezo_30m': 'Город (ИМКЭС, 30 м*)', 
                            'Oblkom_30m': 'Город (Облком)'}
        else:
            site_names2draw = {'BEK_30m':           'Rural (BEK, 30 m)',
                               'Voronino_10m':      'Rural (Voronino)',
                               'IMCES_40m':      'Urban (IMCES, 40 m)', 
                               'IMCES_30m':      'Urban (IMCES, 30 m)',
                               'IMCESmezo_30m': 'Urban (IMCES, 30 m*)', 
                               'Oblkom_30m':        'Urban (Oblkom, 30 m)'}
        

        site_colors = {'BEK_30m': 'blue', 
                       'Voronino_10m': 'deepskyblue',             
                       'IMCES_40m': 'orange', 
                       'IMCES_30m': 'orange',
                       'IMCESmezo_30m': 'orange', 
                       'Oblkom_30m': 'red'}

        site_styles = {'IMCES_30m': '--',
                       'IMCESmezo_30m': ':'}

    site_names2draw ['ERA5'] = 'ERA5'
    site_colors ['ERA5'] = 'black'

    return site_names2draw, site_colors, site_styles