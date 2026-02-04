"""Generate hydrological cluster profiles from physiographic clustering results.

This module synthesizes verbal descriptions blending numerical importance
with hydrological interpretation for each cluster.
"""

import pandas as pd

# Feature descriptions with hydrological interpretations
FEATURE_DESCRIPTIONS = {
    "for_pc_use": {
        "name_ru": "Лесистость",
        "name_en": "Forest Cover",
        "hydrological_impact": "Контролирует эвапотранспирацию через перехват осадков кроной, "
        "стабилизирует почву и способствует инфильтрации. Высокая лесистость "
        "обычно увеличивает базисный сток и снижает пиковые расходы.",
        "low_impact": "Низкая лесистость указывает на открытые территории с быстрым "
        "поверхностным стоком и высокой испаряемостью.",
    },
    "crp_pc_use": {
        "name_ru": "Пахотные земли",
        "name_en": "Cropland",
        "hydrological_impact": "Сельскохозяйственные угодья изменяют инфильтрационную "
        "способность почв, создают сезонную изменчивость шероховатости поверхности. "
        "Уплотнение почв от техники увеличивает поверхностный сток.",
        "low_impact": "Отсутствие пахотных земель указывает на естественный ландшафт "
        "или непригодные для земледелия условия.",
    },
    "inu_pc_ult": {
        "name_ru": "Затапливаемые территории",
        "name_en": "Inundation Area",
        "hydrological_impact": "Пойменные территории обеспечивают естественное регулирование "
        "паводков, аккумулируя воду и замедляя волну половодья. "
        "Способствуют подпитке грунтовых вод.",
        "low_impact": "Отсутствие пойм указывает на врезанные долины с быстрым "
        "транзитом паводковых волн.",
    },
    "ire_pc_use": {
        "name_ru": "Орошаемые территории",
        "name_en": "Irrigated Area",
        "hydrological_impact": "Орошение существенно изменяет водный баланс: увеличивает "
        "водозабор, изменяет режим испарения и может влиять на уровень "
        "грунтовых вод. Требует учета водохозяйственных правил.",
        "low_impact": "Отсутствие орошения указывает на естественный водный режим "
        "или богарное земледелие.",
    },
    "lka_pc_use": {
        "name_ru": "Площадь озер",
        "name_en": "Lake Area",
        "hydrological_impact": "Озера регулируют сток, сглаживая паводковые пики и "
        "поддерживая меженный сток. Увеличивают испарение с водной поверхности.",
        "low_impact": "Отсутствие озер указывает на быструю концентрацию стока "
        "без естественного регулирования.",
    },
    "prm_pc_use": {
        "name_ru": "Мерзлотные породы",
        "name_en": "Permafrost",
        "hydrological_impact": "Многолетняя мерзлота блокирует инфильтрацию, направляя "
        "практически все осадки в поверхностный сток. Создает выраженную "
        "сезонность с экстремальным весенним половодьем.",
        "low_impact": "Отсутствие мерзлоты обеспечивает круглогодичную инфильтрацию "
        "и более равномерное распределение стока.",
    },
    "pst_pc_use": {
        "name_ru": "Пастбища",
        "name_en": "Pasture",
        "hydrological_impact": "Пастбищные угодья характеризуются умеренной инфильтрацией "
        "и шероховатостью. Выпас может уплотнять почву и увеличивать "
        "поверхностный сток.",
        "low_impact": "Отсутствие пастбищ указывает на другие типы землепользования.",
    },
    "cly_pc_uav": {
        "name_ru": "Глинистые почвы",
        "name_en": "Clay Content",
        "hydrological_impact": "Глинистые почвы имеют низкую водопроницаемость, "
        "способствуя поверхностному стоку и образованию верховодки. "
        "Высокая влагоемкость создает память почвенной влаги.",
        "low_impact": "Низкое содержание глины указывает на хорошо дренированные почвы "
        "с быстрой инфильтрацией.",
    },
    "slt_pc_uav": {
        "name_ru": "Суглинистые почвы",
        "name_en": "Silt Content",
        "hydrological_impact": "Суглинки имеют умеренную водопроницаемость, высокую "
        "капиллярную влагоемкость. Склонны к образованию почвенной корки "
        "и эрозии.",
        "low_impact": "Низкое содержание суглинка указывает на преобладание "
        "других фракций (глина или песок).",
    },
    "snd_pc_uav": {
        "name_ru": "Песчаные почвы",
        "name_en": "Sand Content",
        "hydrological_impact": "Песчаные почвы обеспечивают быструю инфильтрацию и "
        "вертикальную фильтрацию, способствуя подпитке грунтовых вод "
        "и стабильному базисному стоку.",
        "low_impact": "Низкое содержание песка указывает на связные почвы "
        "с затрудненной инфильтрацией.",
    },
    "kar_pc_use": {
        "name_ru": "Карстовые породы",
        "name_en": "Karst",
        "hydrological_impact": "Карст создает сложную гидрогеологию с подземным стоком, "
        "родниковым питанием и возможной потерей стока через поноры. "
        "Нелинейная реакция затрудняет моделирование.",
        "low_impact": "Отсутствие карста указывает на более предсказуемую "
        "связь осадки-сток.",
    },
    "urb_pc_use": {
        "name_ru": "Урбанизированные территории",
        "name_en": "Urban Area",
        "hydrological_impact": "Застройка создает непроницаемые поверхности, резко "
        "увеличивая поверхностный сток и пики паводков. Канализация "
        "ускоряет концентрацию стока.",
        "low_impact": "Отсутствие урбанизации указывает на естественные "
        "условия формирования стока.",
    },
    "gwt_cm_sav": {
        "name_ru": "Глубина грунтовых вод",
        "name_en": "Groundwater Table Depth",
        "hydrological_impact": "Глубокое залегание УГВ увеличивает буферную емкость "
        "и время добегания грунтового стока. Поддерживает стабильный "
        "базисный сток в засушливые периоды.",
        "low_impact": "Близкое залегание УГВ указывает на заболоченность "
        "и быструю реакцию на осадки.",
    },
    "lkv_mc_usu": {
        "name_ru": "Объем озер",
        "name_en": "Lake Volume",
        "hydrological_impact": "Большой объем озер обеспечивает значительное регулирование "
        "стока, трансформируя гидрограф и поддерживая меженный сток. "
        "Увеличивает потери на испарение.",
        "low_impact": "Малый объем озер указывает на слабое естественное "
        "регулирование стока.",
    },
    "rev_mc_usu": {
        "name_ru": "Объем водохранилищ",
        "name_en": "Reservoir Volume",
        "hydrological_impact": "Водохранилища полностью трансформируют режим стока "
        "согласно правилам эксплуатации. Требуют данных об управлении "
        "для моделирования.",
        "low_impact": "Отсутствие водохранилищ указывает на естественный "
        "режим стока.",
    },
    "slp_dg_sav": {
        "name_ru": "Средний уклон",
        "name_en": "Mean Slope",
        "hydrological_impact": "Крутые склоны ускоряют концентрацию стока, увеличивают "
        "эрозию и сокращают время добегания. Усиливают пики паводков.",
        "low_impact": "Пологие склоны замедляют сток, способствуют инфильтрации "
        "и аккумуляции воды в понижениях.",
    },
    "sgr_dk_sav": {
        "name_ru": "Градиент уклона",
        "name_en": "Slope Gradient",
        "hydrological_impact": "Неравномерность уклона влияет на распределение "
        "путей стока и зон аккумуляции влаги.",
        "low_impact": "Однородный уклон создает равномерные условия "
        "формирования стока.",
    },
    "ws_area": {
        "name_ru": "Площадь водосбора",
        "name_en": "Watershed Area",
        "hydrological_impact": "Большая площадь увеличивает время концентрации, "
        "сглаживает гидрограф и усредняет пространственную "
        "неоднородность осадков.",
        "low_impact": "Малая площадь создает быструю реакцию на осадки "
        "и локальную изменчивость стока.",
    },
    "ele_mt_uav": {
        "name_ru": "Средняя высота",
        "name_en": "Mean Elevation",
        "hydrological_impact": "Высокогорье увеличивает долю твердых осадков, "
        "создает снеговое питание и выраженное весеннее половодье. "
        "Влияет на температурный градиент и испарение.",
        "low_impact": "Низкие отметки указывают на равнинный характер "
        "с преобладанием дождевого питания.",
    },
}

# Cluster names from radar plots
CLUSTER_NAMES = {
    1: ("Глинистые почвы / Пахотные земли", "Clay soils / Croplands"),
    2: ("Орошаемые территории / Урбанизированные территории", "Irrigated / Urbanized"),
    3: (
        "Средний уклон водосбора / Средняя высота водосбора",
        "Medium slope / Medium elevation",
    ),
    4: ("Мерзлотные породы / Лесистость", "Permafrost / Forested"),
    5: ("Объем озер / Песчаные породы", "Lake volume / Sandy soils"),
    6: ("Затапливаемые территории / Песчаные породы", "Flood-prone / Sandy soils"),
    7: ("Объем озер / Осадочные породы", "Lake volume / Sedimentary"),
    8: ("Карстовые породы / Лесистость", "Karst / Forested"),
    9: ("Лесистость / Осадочные породы", "Forested / Sedimentary"),
    10: ("Песчаные породы / Лесистость", "Sandy soils / Forested"),
}


def interpret_value(value: float) -> str:
    """Интерпретация нормализованного значения (0-1)."""
    if value < 0.2:
        return "Очень низкое"
    elif value < 0.4:
        return "Низкое"
    elif value < 0.6:
        return "Умеренное"
    elif value < 0.8:
        return "Высокое"
    else:
        return "Очень высокое"


def get_runoff_generation(centroid: pd.Series) -> str:
    """Определение механизма формирования стока."""
    clay = centroid.get("cly_pc_uav", 0)
    sand = centroid.get("snd_pc_uav", 0)
    permafrost = centroid.get("prm_pc_use", 0)
    urban = centroid.get("urb_pc_use", 0)
    forest = centroid.get("for_pc_use", 0)
    slope = centroid.get("slp_dg_sav", 0)

    factors = []

    if permafrost > 0.5:
        factors.append(
            "Мерзлота блокирует инфильтрацию, обеспечивая преобладание "
            "поверхностного стока с экстремальной сезонностью."
        )
    elif clay > 0.6 and slope > 0.3:
        factors.append(
            "Сочетание глинистых почв и уклона создает условия для "
            "быстрого поверхностного стока по насыщению."
        )
    elif urban > 0.3:
        factors.append(
            "Урбанизация создает непроницаемые поверхности, "
            "резко увеличивая поверхностный сток."
        )
    elif sand > 0.6 and forest > 0.5:
        factors.append(
            "Песчаные почвы под лесом обеспечивают преобладание "
            "инфильтрации и подземного питания."
        )
    elif sand > 0.5:
        factors.append(
            "Песчаные почвы способствуют инфильтрации "
            "и подпитке грунтовых вод."
        )
    elif clay > 0.5:
        factors.append(
            "Глинистые почвы ограничивают инфильтрацию, "
            "формируя смешанный тип питания с преобладанием поверхностного."
        )
    else:
        factors.append(
            "Смешанный тип формирования стока с балансом "
            "поверхностной и подземной составляющих."
        )

    return " ".join(factors)


def get_flow_regime(centroid: pd.Series) -> str:
    """Определение режима стока."""
    lake_vol = centroid.get("lkv_mc_usu", 0)
    lake_area = centroid.get("lka_pc_use", 0)
    reservoir = centroid.get("rev_mc_usu", 0)
    permafrost = centroid.get("prm_pc_use", 0)
    slope = centroid.get("slp_dg_sav", 0)
    inundation = centroid.get("inu_pc_ult", 0)
    elevation = centroid.get("ele_mt_uav", 0)

    factors = []

    if reservoir > 0.3:
        factors.append(
            "Режим трансформирован водохранилищами — сглаженный гидрограф, "
            "зависимость от правил управления."
        )
    elif lake_vol > 0.5 or lake_area > 0.5:
        factors.append(
            "Озерное регулирование сглаживает паводки и поддерживает "
            "стабильный меженный сток."
        )
    elif permafrost > 0.5 and elevation > 0.3:
        factors.append(
            "Выраженная сезонность: экстремальное весеннее половодье "
            "при таянии на мерзлоте, низкая межень."
        )
    elif slope > 0.4:
        factors.append(
            "Быстрая реакция на осадки, пикообразный гидрограф "
            "с коротким временем добегания."
        )
    elif inundation > 0.3:
        factors.append(
            "Пойменное регулирование замедляет паводковую волну "
            "и распластывает гидрограф."
        )
    else:
        factors.append(
            "Умеренная реакция на осадки с постепенным подъемом "
            "и спадом гидрографа."
        )

    return " ".join(factors)


def get_baseflow(centroid: pd.Series) -> str:
    """Оценка базисного стока."""
    sand = centroid.get("snd_pc_uav", 0)
    karst = centroid.get("kar_pc_use", 0)
    gwt = centroid.get("gwt_cm_sav", 0)
    forest = centroid.get("for_pc_use", 0)
    lake_vol = centroid.get("lkv_mc_usu", 0)
    permafrost = centroid.get("prm_pc_use", 0)

    if permafrost > 0.5:
        return (
            "**Низкий**. Мерзлота блокирует подземное питание, "
            "базисный сток минимален и формируется только в деятельном слое."
        )
    elif karst > 0.5:
        return (
            "**Высокий**. Карстовые породы обеспечивают устойчивое "
            "родниковое питание с минимальной сезонной изменчивостью."
        )
    elif sand > 0.6 and gwt > 0.3:
        return (
            "**Высокий**. Песчаные почвы и глубокий УГВ создают "
            "значительный буфер грунтовых вод для стабильного базисного стока."
        )
    elif lake_vol > 0.4:
        return (
            "**Умеренно высокий**. Озерная аккумуляция поддерживает "
            "сток в меженные периоды."
        )
    elif forest > 0.7 and sand > 0.4:
        return (
            "**Умеренно высокий**. Лесной покров с проницаемыми почвами "
            "способствует инфильтрации и устойчивому подземному питанию."
        )
    elif sand > 0.5:
        return (
            "**Умеренный**. Песчаные почвы обеспечивают подпитку грунтовых вод, "
            "но объем буфера ограничен."
        )
    else:
        return (
            "**Низкий до умеренного**. Связные почвы ограничивают инфильтрацию, "
            "базисный сток зависит от локальных условий."
        )


def get_flood_response(centroid: pd.Series) -> str:
    """Оценка реакции на паводки."""
    slope = centroid.get("slp_dg_sav", 0)
    lake_vol = centroid.get("lkv_mc_usu", 0)
    inundation = centroid.get("inu_pc_ult", 0)
    urban = centroid.get("urb_pc_use", 0)
    reservoir = centroid.get("rev_mc_usu", 0)
    area = centroid.get("ws_area", 0)
    permafrost = centroid.get("prm_pc_use", 0)

    if reservoir > 0.3:
        return (
            "**Зарегулированная**. Водохранилища срезают пики, "
            "но эффективность зависит от наполнения и режима сбросов."
        )
    elif lake_vol > 0.5 or inundation > 0.4:
        return (
            "**Сглаженная**. Естественное регулирование озерами и поймами "
            "трансформирует паводковую волну, снижая пики."
        )
    elif urban > 0.4 and slope > 0.2:
        return (
            "**Очень быстрая**. Урбанизация и уклон создают "
            "экстремально короткое время концентрации и высокие пики."
        )
    elif permafrost > 0.5:
        return (
            "**Быстрая и экстремальная**. Весеннее половодье на мерзлоте "
            "формирует максимальные расходы года за короткий период."
        )
    elif slope > 0.4:
        return (
            "**Быстрая**. Крутые склоны обеспечивают малое время добегания "
            "и острые пики гидрографа."
        )
    elif area > 0.5:
        return (
            "**Замедленная**. Большая площадь водосбора увеличивает время "
            "концентрации и сглаживает пиковые расходы."
        )
    else:
        return (
            "**Умеренная**. Типичная реакция с постепенным формированием "
            "и прохождением паводка."
        )


def get_drought_resilience(centroid: pd.Series) -> str:
    """Оценка устойчивости к засухам."""
    gwt = centroid.get("gwt_cm_sav", 0)
    lake_vol = centroid.get("lkv_mc_usu", 0)
    forest = centroid.get("for_pc_use", 0)
    sand = centroid.get("snd_pc_uav", 0)
    karst = centroid.get("kar_pc_use", 0)
    permafrost = centroid.get("prm_pc_use", 0)

    if karst > 0.5 and gwt > 0.2:
        return (
            "**Высокая**. Карстовые водоносные горизонты обеспечивают "
            "устойчивое родниковое питание даже в засушливые периоды."
        )
    elif lake_vol > 0.5:
        return (
            "**Высокая**. Значительный объем озер служит буфером, "
            "поддерживая сток при дефиците осадков."
        )
    elif gwt > 0.4 and sand > 0.5:
        return (
            "**Умеренно высокая**. Запасы грунтовых вод в проницаемых породах "
            "поддерживают базисный сток."
        )
    elif forest > 0.7:
        return (
            "**Умеренная**. Лес буферизирует испарение, но при длительной "
            "засухе транспирация истощает почвенную влагу."
        )
    elif permafrost > 0.5:
        return (
            "**Низкая**. Отсутствие значимых подземных запасов делает сток "
            "полностью зависимым от текущих осадков и снеготаяния."
        )
    else:
        return (
            "**Умеренная до низкой**. Устойчивость определяется локальными "
            "запасами почвенной влаги и грунтовых вод."
        )


def get_anthropogenic_alteration(centroid: pd.Series) -> str:
    """Оценка антропогенной трансформации."""
    irrigation = centroid.get("ire_pc_use", 0)
    urban = centroid.get("urb_pc_use", 0)
    reservoir = centroid.get("rev_mc_usu", 0)
    cropland = centroid.get("crp_pc_use", 0)

    factors = []

    if reservoir > 0.2:
        factors.append(f"водохранилища ({interpret_value(reservoir).lower()})")
    if irrigation > 0.2:
        factors.append(f"орошение ({interpret_value(irrigation).lower()})")
    if urban > 0.2:
        factors.append(f"урбанизация ({interpret_value(urban).lower()})")
    if cropland > 0.3:
        factors.append(f"распашка ({interpret_value(cropland).lower()})")

    if not factors:
        return "**Минимальная**. Водосборы сохраняют близкий к естественному режим."
    elif len(factors) == 1:
        return f"**Умеренная**: {factors[0]}."
    else:
        return f"**Значительная**: {', '.join(factors)}."


def get_model_expectations(centroid: pd.Series) -> list[tuple[str, str, str]]:
    """Прогноз поведения моделей."""
    karst = centroid.get("kar_pc_use", 0)
    permafrost = centroid.get("prm_pc_use", 0)
    irrigation = centroid.get("ire_pc_use", 0)
    reservoir = centroid.get("rev_mc_usu", 0)
    urban = centroid.get("urb_pc_use", 0)
    lake_vol = centroid.get("lkv_mc_usu", 0)

    results = []

    # HBV
    if permafrost > 0.5:
        hbv = ("Хорошая", "Снеговая рутина HBV хорошо описывает мерзлотную динамику.")
    elif karst > 0.5:
        hbv = ("Умеренная", "Линейные резервуары не передают карстовую нелинейность.")
    elif reservoir > 0.3 or irrigation > 0.3:
        hbv = ("Слабая", "Нет модуля управления водохозяйственными объектами.")
    else:
        hbv = ("Хорошая", "Стандартные процессы хорошо параметризуются.")
    results.append(("HBV", *hbv))

    # GR4J
    if permafrost > 0.5:
        gr4j = ("Умеренная", "Упрощенная снеговая схема CemaNeige для мерзлоты.")
    elif lake_vol > 0.5:
        gr4j = ("Хорошая", "Резервуарная структура адаптируется к озерному регулированию.")
    elif karst > 0.5:
        gr4j = ("Слабая", "4 параметра недостаточны для карстовой сложности.")
    else:
        gr4j = ("Хорошая", "Парсимоничная структура эффективна для типичных водосборов.")
    results.append(("GR4J", *gr4j))

    # LSTM
    if reservoir > 0.3 or irrigation > 0.3:
        lstm = ("Умеренная", "Требует данных об управлении как входных признаков.")
    elif karst > 0.5:
        lstm = ("Хорошая", "Способна выучить нелинейные паттерны при достатке данных.")
    else:
        lstm = ("Хорошая", "Эффективно выявляет сложные зависимости осадки-сток.")
    results.append(("LSTM", *lstm))

    return results


def get_calibration_considerations(centroid: pd.Series) -> list[str]:
    """Ключевые аспекты калибровки."""
    considerations = []

    karst = centroid.get("kar_pc_use", 0)
    permafrost = centroid.get("prm_pc_use", 0)
    irrigation = centroid.get("ire_pc_use", 0)
    reservoir = centroid.get("rev_mc_usu", 0)
    lake_vol = centroid.get("lkv_mc_usu", 0)
    urban = centroid.get("urb_pc_use", 0)
    slope = centroid.get("slp_dg_sav", 0)
    inundation = centroid.get("inu_pc_ult", 0)

    if permafrost > 0.4:
        considerations.append(
            "Параметры снеготаяния критичны — температурный индекс и "
            "коэффициенты таяния требуют тщательной настройки."
        )
    if karst > 0.4:
        considerations.append(
            "Двойная пористость карста требует разделения быстрого и "
            "медленного подземного стока."
        )
    if reservoir > 0.2 or irrigation > 0.2:
        considerations.append(
            "Необходимы данные о водозаборе/сбросах или отдельный модуль "
            "водохозяйственного управления."
        )
    if lake_vol > 0.4:
        considerations.append(
            "Параметры озерного регулирования (коэффициент истощения) "
            "существенно влияют на меженный сток."
        )
    if urban > 0.3:
        considerations.append(
            "Доля непроницаемых поверхностей и параметры канализации "
            "определяют пиковый отклик."
        )
    if slope > 0.4:
        considerations.append(
            "Время концентрации и параметры маршрутизации требуют "
            "калибровки по форме гидрографа."
        )
    if inundation > 0.3:
        considerations.append(
            "Пойменная аккумуляция требует калибровки порога затопления "
            "и коэффициента трансформации."
        )

    if not considerations:
        considerations.append(
            "Стандартная калибровка по KGE с акцентом на баланс "
            "пиковых и меженных расходов."
        )

    return considerations


def generate_cluster_profile(
    cluster_id: int,
    centroid: pd.Series,
    n_catchments: int,
) -> str:
    """Генерация профиля кластера в формате Markdown.

    Args:
        cluster_id: Идентификатор кластера (1-10).
        centroid: Нормализованные значения признаков (центроид).
        n_catchments: Количество водосборов в кластере.

    Returns:
        Markdown-строка с профилем кластера.
    """
    name_ru, name_en = CLUSTER_NAMES.get(cluster_id, ("Без названия", "Unnamed"))

    # Exclude non-feature columns
    feature_cols = [c for c in centroid.index if c not in ["cluster_geo", "gauge_id"]]
    features = centroid[feature_cols].sort_values(ascending=False)

    # Top 3 dominant features
    dominant = features.head(3)

    # Bottom 3 deficient features
    deficient = features.tail(3).sort_values()

    lines = [
        f"## Ф{cluster_id}: {name_ru} / {name_en}",
        f"**n = {n_catchments} водосборов**",
        "",
        "### Доминирующие признаки",
        "| Признак | Значение | Интерпретация |",
        "|---------|----------|---------------|",
    ]

    for feat, val in dominant.items():
        desc = FEATURE_DESCRIPTIONS.get(feat, {})
        name = desc.get("name_ru", feat)
        impact = desc.get("hydrological_impact", "—")
        lines.append(f"| {name} | {val:.2f} | {impact} |")

    lines.extend(
        [
            "",
            "### Дефицитные признаки",
            "| Признак | Значение | Следствие |",
            "|---------|----------|-----------|",
        ]
    )

    for feat, val in deficient.items():
        desc = FEATURE_DESCRIPTIONS.get(feat, {})
        name = desc.get("name_ru", feat)
        impact = desc.get("low_impact", "—")
        lines.append(f"| {name} | {val:.2f} | {impact} |")

    lines.extend(
        [
            "",
            "### Синтез гидрологического режима",
            "",
            "**Формирование стока**:",
            get_runoff_generation(centroid),
            "",
            "**Режим стока**:",
            get_flow_regime(centroid),
            "",
            "**Базисный сток**:",
            get_baseflow(centroid),
            "",
            "**Реакция на паводки**:",
            get_flood_response(centroid),
            "",
            "**Устойчивость к засухам**:",
            get_drought_resilience(centroid),
            "",
            "**Антропогенная трансформация**:",
            get_anthropogenic_alteration(centroid),
            "",
            "### Ожидаемое поведение моделей",
            "",
            "| Модель | Прогноз | Обоснование |",
            "|--------|---------|-------------|",
        ]
    )

    for model, perf, reason in get_model_expectations(centroid):
        lines.append(f"| {model} | {perf} | {reason} |")

    lines.extend(
        [
            "",
            "### Ключевые аспекты калибровки",
            "",
        ]
    )

    for consideration in get_calibration_considerations(centroid):
        lines.append(f"- {consideration}")

    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def generate_summary_table(cluster_stats: dict) -> str:
    """Генерация сводной таблицы кластеров.

    Args:
        cluster_stats: Словарь {cluster_id: {'centroid': Series, 'n': int}}.

    Returns:
        Markdown-таблица.
    """
    lines = [
        "## Сводная таблица кластеров",
        "",
        "| Кластер | n | Доминирующий процесс | Режим стока | Сложность моделирования |",
        "|---------|---|---------------------|-------------|------------------------|",
    ]

    for cid in sorted(cluster_stats.keys()):
        stats = cluster_stats[cid]
        centroid = stats["centroid"]
        n = stats["n"]

        # Determine dominant process
        permafrost = centroid.get("prm_pc_use", 0)
        karst = centroid.get("kar_pc_use", 0)
        urban = centroid.get("urb_pc_use", 0)
        irrigation = centroid.get("ire_pc_use", 0)
        lake_vol = centroid.get("lkv_mc_usu", 0)
        sand = centroid.get("snd_pc_uav", 0)
        clay = centroid.get("cly_pc_uav", 0)

        if permafrost > 0.5:
            process = "Мерзлотный сток"
        elif karst > 0.5:
            process = "Карстовый сток"
        elif urban > 0.3 or irrigation > 0.3:
            process = "Антропогенный"
        elif lake_vol > 0.4:
            process = "Озерное регулирование"
        elif sand > 0.6:
            process = "Инфильтрационный"
        elif clay > 0.6:
            process = "Поверхностный сток"
        else:
            process = "Смешанный"

        # Flow regime
        slope = centroid.get("slp_dg_sav", 0)
        reservoir = centroid.get("rev_mc_usu", 0)

        if reservoir > 0.2:
            regime = "Зарегулированный"
        elif permafrost > 0.5:
            regime = "Сезонный экстремальный"
        elif lake_vol > 0.4:
            regime = "Сглаженный"
        elif slope > 0.3:
            regime = "Пикообразный"
        else:
            regime = "Умеренный"

        # Modeling challenge
        challenges = []
        if karst > 0.4:
            challenges.append("карст")
        if permafrost > 0.4:
            challenges.append("мерзлота")
        if irrigation > 0.2 or reservoir > 0.2:
            challenges.append("управление")
        if urban > 0.3:
            challenges.append("урбанизация")

        challenge = ", ".join(challenges) if challenges else "Стандартная"

        name_ru, _ = CLUSTER_NAMES.get(cid, ("—", "—"))
        lines.append(f"| Ф{cid} | {n} | {process} | {regime} | {challenge} |")

    return "\n".join(lines)


def generate_model_suitability_matrix(cluster_stats: dict) -> str:
    """Генерация матрицы пригодности моделей.

    Args:
        cluster_stats: Словарь {cluster_id: {'centroid': Series, 'n': int}}.

    Returns:
        Markdown-таблица.
    """
    lines = [
        "## Матрица пригодности моделей",
        "",
        "| Кластер | HBV | GR4J | LSTM | Лучший выбор |",
        "|---------|-----|------|------|--------------|",
    ]

    for cid in sorted(cluster_stats.keys()):
        centroid = cluster_stats[cid]["centroid"]
        expectations = get_model_expectations(centroid)

        scores = {}
        for model, perf, _ in expectations:
            if perf == "Хорошая":
                scores[model] = "++"
            elif perf == "Умеренная":
                scores[model] = "+"
            else:
                scores[model] = "−"

        # Determine best choice
        best_scores = [
            (m, 2 if s == "++" else 1 if s == "+" else 0) for m, s in scores.items()
        ]
        best = max(best_scores, key=lambda x: x[1])[0]

        lines.append(
            f"| Ф{cid} | {scores.get('HBV', '?')} | {scores.get('GR4J', '?')} | "
            f"{scores.get('LSTM', '?')} | {best} |"
        )

    lines.extend(
        [
            "",
            "*Обозначения: ++ хорошая, + умеренная, − слабая*",
        ]
    )

    return "\n".join(lines)


def compute_cluster_centroids(df: pd.DataFrame) -> dict:
    """Вычисление центроидов кластеров.

    Args:
        df: DataFrame с признаками и столбцом cluster_geo.

    Returns:
        Словарь {cluster_id: {'centroid': Series, 'n': int}}.
    """
    feature_cols = [c for c in df.columns if c not in ["gauge_id", "cluster_geo"]]

    result = {}
    for cid in sorted(df["cluster_geo"].unique()):
        cluster_df = df[df["cluster_geo"] == cid]
        centroid = cluster_df[feature_cols].mean()
        result[cid] = {"centroid": centroid, "n": len(cluster_df)}

    return result


def generate_all_profiles(csv_path: str, output_path: str) -> None:
    """Генерация полного документа профилей кластеров.

    Args:
        csv_path: Путь к CSV с данными (geo_scaled.csv).
        output_path: Путь для сохранения Markdown.
    """
    from src.utils.logger import setup_logger

    logger = setup_logger("cluster_profiles")
    logger.info(f"Загрузка данных из {csv_path}")

    df = pd.read_csv(csv_path)
    cluster_stats = compute_cluster_centroids(df)

    lines = [
        "# Профили физиографических кластеров",
        "",
        "Синтез гидрологических характеристик на основе нормализованных "
        "значений признаков (шкала 0–1).",
        "",
    ]

    # Summary tables first
    lines.append(generate_summary_table(cluster_stats))
    lines.extend(["", ""])
    lines.append(generate_model_suitability_matrix(cluster_stats))
    lines.extend(["", "---", ""])

    # Individual profiles
    for cid in sorted(cluster_stats.keys()):
        logger.info(f"Генерация профиля кластера Ф{cid}")
        profile = generate_cluster_profile(
            cluster_id=cid,
            centroid=cluster_stats[cid]["centroid"],
            n_catchments=cluster_stats[cid]["n"],
        )
        lines.append(profile)

    content = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Профили сохранены в {output_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Default paths
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / "res/chapter_one/data/geo_scaled.csv"
    output_path = project_root / "res/chapter_one/tables/cluster_profiles.md"

    # Allow override via command line
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_all_profiles(str(csv_path), str(output_path))
