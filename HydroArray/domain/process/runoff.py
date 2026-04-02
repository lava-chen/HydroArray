"""
产流计算

提供蓄满产流模型计算功能。
"""

import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

from HydroArray.domain.process.evaporation import three_layer_evaporation


def _round_half_up(value: float, decimals: int = 1) -> float:
    """
    标准四舍五入（ROUND_HALF_UP），区别于 Python 内置的银行家舍入

    Args:
        value: 要舍入的值
        decimals: 保留的小数位数，默认 1

    Returns:
        float: 舍入后的值
    """
    d = Decimal(str(value))
    rounded = d.quantize(Decimal(10) ** -decimals, rounding=ROUND_HALF_UP)
    return float(rounded)


def saturation_excess_runoff(
    data_df: pd.DataFrame,
    WUM: float,
    WLM: float,
    WDM: float,
    C: float,
    Kc: float,
    b: float,
    initial_WU: float,
    initial_WL: float,
    initial_WD: float,
    initial_R: float,
    initial_a: float | None = None
) -> pd.DataFrame:
    """
    蓄满产流模型计算，使用三层蒸发模型计算流域蒸散发量，并结合蓄满产流模型计算产流量。

    Args:
        data_df: 包含 'date', 'P', 'Ep' 列的数据框
        WUM: 上层土壤含水容量 (mm)
        WLM: 下层土壤含水容量 (mm)
        WDM: 深层土壤含水容量 (mm)
        C: 蒸发扩散系数
        Kc: 蒸发折算系数
        b: 蓄满产流参数，反映流域包气带蓄水容量分布的不均匀性
        initial_WU: 初始上层土壤含水量 (mm)
        initial_WL: 初始下层土壤含水量 (mm)
        initial_WD: 初始深层土壤含水量 (mm)
        initial_R: 初始产流量 (mm)
        initial_a: 初始蓄水容量曲线参数

    Returns:
        pd.DataFrame: 包含以下列的计算结果：
            - date: 日期
            - P: 降雨量 (mm)
            - Ep: 蒸发能力 (mm)
            - EU: 上层蒸发量 (mm)
            - EL: 下层蒸发量 (mm)
            - ED: 深层蒸发量 (mm)
            - E: 总蒸发量 (mm)
            - PE: 净雨量 (mm)
            - WU: 上层土壤含水量 (mm)
            - WL: 下层土壤含水量 (mm)
            - WD: 深层土壤含水量 (mm)
            - W: 总土壤含水量 (mm)
            - R: 产流量 (mm)
    """
    results = []
    WU = initial_WU
    WL = initial_WL
    WD = initial_WD
    R = initial_R
    W_total = WU + WL + WD
    WM = WUM + WLM + WDM
    WMM = (1 + b) * WM
    if initial_a is not None:
        a = initial_a
    else:
        a = WMM * ( 1 - ( 1 - W_total / WM) ** (1 + b))

    for _, row in data_df.iterrows():
        WU_0, WL_0, WD_0 = WU, WL, WD
        W_total_0 = W_total
        date = row['date']
        P = row['P'] if pd.notna(row['P']) else 0.0
        alpha = 1 - (1 - a/WMM) ** b
        if 'Ep' in row:
            Ep = row['Ep'] if pd.notna(row['Ep']) else 0.0
        elif 'E0' in row:
            E0 = row['E0'] if pd.notna(row['E0']) else 0.0
            Ep = Kc * E0
        else:
            E0 = row['E'] if pd.notna(row['E']) else 0.0
            Ep = Kc * E0

        EU, EL, ED = three_layer_evaporation(Ep, WU, WL, P, WLM, C)
        EU_0, EL_0, ED_0 = EU, EL, ED
        E_total = EU + EL + ED

        # 中间计算保留高精度，最后输出时再舍入
        PE = P - E_total

        if PE < 0:
            deficit = P
            deduct = min(EU, deficit)
            EU -= deduct
            deficit -= deduct

            deduct = min(EL, deficit)
            EL -= deduct
            deficit -= deduct

            ED = max(0, ED - deficit)

            WU = WU - EU
            WL = WL - EL
            WD = WD - ED

            W_total = WU + WL + WD

            # 蒸发后根据新的土壤含水量重新计算a
            a = WMM * (1 - (1 - min(W_total, WM) / WM) ** (1 + b))
            a = max(0, min(a, WMM))  # 限制a在合理范围

            R = 0.0

        elif PE > 0:
            if a + PE <= WMM:
                R = PE + W_total - WM + WM * (1 - (PE + a) / WMM) ** (b + 1)
                R = max(R, 0)  # 确保产流量不为负
                if a + PE > WMM:
                    a = WMM
                else:
                    a = a + PE
            else:
                R = max(PE - WM + W_total, 0)

            surplus = PE - R
            add = min(WUM - WU, surplus)
            WU += add
            surplus -= add

            add = min(WLM - WL, surplus)
            WL += add
            surplus -= add

            WD = min(WDM, WD + surplus)

            W_total = WU + WL + WD

        results.append({
            'date': date,
            'P': P,
            'Ep': Ep,
            'EU': EU_0,
            'EL': EL_0,
            'ED': ED_0,
            'E': E_total,
            'PE': PE,
            'WU': WU_0,
            'WL': WL_0,
            'WD': WD_0,
            'W': W_total_0,
            'R': R,
            'a': a,
            'alpha': alpha
        })

    result_df = pd.DataFrame(results)

    round_cols = ['EU', 'EL', 'ED', 'E', 'PE', 'WU', 'WL', 'WD', 'W', 'R', 'a', 'alpha']
    for col in round_cols:
        if col in result_df.columns:
            result_df[col] = result_df[col].apply(lambda x: _round_half_up(x, 1))

    return result_df


def two_source_runoff_separation(
    data_df: pd.DataFrame,
    FC: float
) -> pd.DataFrame:
    """
    两水源划分 - 通过渗透率将总产流量划分为地表径流和直接径流。
        PE < FC:
            RD = 0
            RG = PE * alpha
        PE > FC:
            RG = FC * R / PE
            RD = R - RG

    Args:
        data_df: 包含 'date', 'R', 'PE', 'alpha' 列的数据框（可来自 saturation_excess_runoff 输出）
        FC: 实际稳渗值 (mm/d)

    Returns:
        pd.DataFrame: 新增以下列的计算结果：
            - RG: 地下径流 (mm)
            - RD: 直接径流 (mm)
    """
    results = []

    for _, row in data_df.iterrows():
        R = row['R'] if pd.notna(row['R']) else 0.0
        PE= row['PE'] if pd.notna(row['PE']) else 0.0
        alpha = row['alpha'] if pd.notna(row['alpha']) else 0.0
        if R > 0:
            if PE <= FC:
                RD = 0.0
                RG = PE * alpha
            elif PE > FC:
                RG = FC * R / PE
                RD = R - RG
        else:
            RD = 0.0
            RG = 0.0
        results.append({
            'date': row['date'],
            'RD': RD,
            'RG': RG
        })

    result_df = pd.DataFrame(results)

    # 输出时对 RD, RG 保留1位小数
    result_df['RD'] = result_df['RD'].apply(lambda x: _round_half_up(x, 1))
    result_df['RG'] = result_df['RG'].apply(lambda x: _round_half_up(x, 1))

    return result_df

def three_source_runoff_separation(
    data_df: pd.DataFrame,
    SM: float,
    EX: float,
    KI: float,
    KG: float,
    initial_S: float
):
    """
    三水源划分 - 利用水箱概念模型将水源划分为地面径流、壤中径流和地下径流
        与蓄满产流模型类似，由于下垫面的不均匀性，自由蓄水量S也存在空间分布不均匀性。分布特征采用指数方程近似描述:
        alpha = 1 - (1 - S/SMM) ** EX
        SMM = SM * (1 + EX)
        AU = SMM * (1 - (1 - (S1 * FR1)/(SM * FR)) ** (1 / (1 + EX)))
        FR = 1 - (1 - S1 / SMM) ** EX
        PE + AU < SMM:
            RS = FR * (PE + S1 * FR1 / FR - SM + SM * (1 - (PE + AU) / SMM) ** (EX + 1))
        PE + AU >= SMM:
            RS = FR * (PE + S1 * FR1 / FR - SM)
            S = S1 * FR1 / FR + (R - RS) / FR
            RI = KI * S * FR
            RG = KG * S * FR
            S1 = S * (1 - KI - KG)
        
    Args:
        data_df:
        SM: 自由水蓄水容量(mm)
        EX: 自由水蓄量分布
        KI: 壤中流出流系数
        KG: 地下径流出流系数
        initial_S: 初始自由水蓄量(mm)

    Returns:
        pd.DataFrame: 新增以下列的计算结果：
            - RS: 地面径流 (mm)
            - RI: 壤中流 (mm)
            - RG: 地下径流 (mm)
    """
    results = []
    S1 = initial_S
    FR1 = 1.0
    SMM = SM * (1 + EX)

    for _, row in data_df.iterrows():
        R = row['R'] if pd.notna(row['R']) else 0.0
        PE = row['PE'] if pd.notna(row['PE']) else 0.0

        if R > 0:
            FR = 1 - (1 - S1 / SMM) ** EX if SMM > 0 else 0
            FR = max(FR, 1e-10)

            ratio = (S1 * FR1) / (SM * FR) if SM * FR > 0 else 0
            ratio = min(ratio, 1.0)
            AU = SMM * (1 - (1 - ratio) ** (1 / (1 + EX)))

            if PE + AU < SMM:
                RS = FR * (PE + S1 * FR1 / FR - SM + SM * (1 - (PE + AU) / SMM) ** (EX + 1))
                RS = max(RS, 0)
            else:
                RS = FR * (PE + S1 * FR1 / FR - SM)
                RS = max(RS, 0)

            S = S1 * FR1 / FR + (R - RS) / FR
            S = min(S, SM)

            RI = KI * S * FR
            RG = KG * S * FR

            S1 = S * (1 - KI - KG)
            FR1 = FR
        else:
            RS = 0.0
            RI = 0.0
            RG = 0.0

        results.append({
            'date': row['date'],
            'RS': RS,
            'RI': RI,
            'RG': RG
        })

    result_df = pd.DataFrame(results)

    result_df['RS'] = result_df['RS'].apply(lambda x: _round_half_up(x, 1))
    result_df['RI'] = result_df['RI'].apply(lambda x: _round_half_up(x, 1))
    result_df['RG'] = result_df['RG'].apply(lambda x: _round_half_up(x, 1))

    return result_df