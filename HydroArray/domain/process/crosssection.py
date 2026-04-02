"""
断面计算

提供大断面和水道断面的计算功能，遵循水文测验规范进行数值修约。
"""

import numpy as np
import pandas as pd

from HydroArray.utils.rounding import round_to_n_sig_figs, round_area, round_distance, round_width

def calculate_cross_section_area(distances, elevations, target_z_seq):
    """
    使用矩阵直接计算断面过水面积，简洁版本，避免循环计算
    
    Args:
        distances: 起点距列表 (x坐标)
        elevations: 地形高程列表 (z坐标)
        target_z_seq: 目标水位序列
    
    Returns:
        list: 包含 (水位, 过水面积) 元组的列表

    Example:
        >>> distances = [0, 1, 2, 3, 4]
        >>> elevations = [0, 1, 2, 3, 4]
        >>> target_z_seq = [2.5, 3.5]
        >>> areas = calculate_cross_section_area(distances, elevations, target_z_seq)
    """
    x = np.array(distances)
    z = np.array(elevations)
    
    Z = np.array(target_z_seq)[:, np.newaxis] # (M, 1)
    
    dx = np.diff(x)  # (N-1,)
    
    h1 = Z - z[:-1]  # (M, N-1)
    h2 = Z - z[1:]   # (M, N-1)
    
    areas = np.zeros_like(h1) # (M, N-1)
    dx_b = np.broadcast_to(dx, h1.shape) # (M, N-1)
    
    # 掩码提取 3 种有效情况，直接矩阵内相加
    # 1. 梯形 (全水下)
    mask_both = (h1 >= 0) & (h2 >= 0)
    areas[mask_both] = 0.5 * dx_b[mask_both] * (h1[mask_both] + h2[mask_both])
    
    # 2. 左三角形 (跨越)
    mask_left = (h1 >= 0) & (h2 < 0)
    areas[mask_left] = 0.5 * dx_b[mask_left] * (h1[mask_left]**2) / (h1[mask_left] - h2[mask_left])
    
    # 3. 右三角形 (跨越)
    mask_right = (h1 < 0) & (h2 >= 0)
    areas[mask_right] = 0.5 * dx_b[mask_right] * (h2[mask_right]**2) / (h2[mask_right] - h1[mask_right])
    
    areas = np.sum(areas, axis=1)

    return list(zip(target_z_seq, areas))

def calculate_channel_section_detailed(
        data_df: pd.DataFrame,
        lowest_elevation: float
    ) -> tuple[pd.DataFrame, float]:
    """
    水道断面计算

    Args:
        data_df: 包含起点距（默认左岸）和各测点河底高程的DataFrame
        lowest_elevation: 历年最低水位
    
    Returns:
        tuple: (水道断面计算表格DataFrame, 水道断面面积)
    """
    x = data_df.iloc[:, 1].values.astype(float)
    z = data_df.iloc[:, 2].values.astype(float)

    insert_points = []
    for i in range(len(x) - 1):
        z1, z2 = z[i], z[i+1]
        if (z1 < lowest_elevation and z2 > lowest_elevation) or (z1 > lowest_elevation and z2 < lowest_elevation):
            x_int = x[i] + (x[i+1] - x[i]) * (lowest_elevation - z1) / (z2 - z1)
            x_int = round_distance(x_int)
            insert_points.append((i + 1, x_int, lowest_elevation))
    
    if insert_points:
        new_x, new_z = [x[0]], [z[0]]
        for i in range(len(x) - 1):
            for idx, x_val, z_val in insert_points:
                if idx == i + 1:
                    new_x.append(x_val)
                    new_z.append(z_val)
            new_x.append(x[i + 1])
            new_z.append(z[i + 1])
        x = np.array(new_x)
        z = np.array(new_z)
    
    mask = z <= lowest_elevation
    x = x[mask]
    z = z[mask]

    h = lowest_elevation - z

    avg_h = []
    width = []
    area_delta = []
    for i in range(len(h)-1):
        avg_h.append((h[i] + h[i+1]) / 2)
        width.append(round_width(x[i+1] - x[i]))
        area_delta.append(((h[i+1] + h[i]) / 2) * (x[i+1] - x[i]))

    result = []
    area = 0
    for i in range(len(h)):
        result.append({
            'z': z[i],
            'x': round_distance(x[i]),
            'h': round_to_n_sig_figs(h[i], 3),
            'avg_h': round_to_n_sig_figs(avg_h[i-1], 3) if i-1 >= 0 else None,
            'width': round_width(width[i-1]) if i-1 >= 0 else None,
            'area_delta': round_area(area_delta[i-1]) if i-1 >= 0 else None,
            'area': area
        })
        if i < len(area_delta):
            area += area_delta[i]
            area = round_area(area)
    
    return pd.DataFrame(result), area

def calculate_cross_section_area_detailed(
        data_df: pd.DataFrame,
        given_elevation_list: list,
        lowest_elevation: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    大断面计算，根据断面数据（包括：起点距（默认左岸），和各测点的河底高程）

    Args:
        data_df: 包含起点距（默认左岸）和各测点河底高程的DataFrame
        given_elevation_list: 给定的排序河底高程列表
        lowest_elevation: 历年最低水位
    
    Returns:
        tuple: (大断面计算结果DataFrame, 水道断面计算结果DataFrame)
    """
    x = data_df.iloc[:, 1].values.astype(float)
    z = data_df.iloc[:, 2].values.astype(float)
    
    result = []
    cumulative_area = 0.0

    # 假设你外部已经有这个函数的实现
    channel_section_result, channel_area = calculate_channel_section_detailed(data_df, lowest_elevation)
    
    for i, Z in enumerate(given_elevation_list):
        intersections = []
        
        # 1. 遍历所有线段，寻找与当前水位 Z 的交点
        for j in range(len(x) - 1):
            x1, z1, x2, z2 = x[j], z[j], x[j+1], z[j+1]
            
            # 刚好等于的情况（把水下的实际测点也收录进来）
            if z1 == Z: intersections.append(x1)
            if z2 == Z: intersections.append(x2)
            
            # 严格相交的情况（线段跨越了水位面），用相似三角形求交点 x
            if (z1 > Z and z2 < Z) or (z1 < Z and z2 > Z):
                x_int = x1 + (x2 - x1) * (Z - z1) / (z2 - z1)
                intersections.append(x_int)
                
            # 如果整条线段都在水底，把它的两个端点加进去
            # 这样可以在后面排序去重时，保留水底的真实宽度
            if z1 < Z and z2 < Z:
                intersections.append(x1)
                intersections.append(x2)
                
        # 2. 去重并从小到大排序
        # 必须舍入到一个合理的精度以避免浮点数精度带来的重复点
        intersections = np.round(intersections, 4) 
        unique_x_points = sorted(list(set(intersections)))
        
        net_width = 0.0
        L_dist = None
        R_dist = None
        
        if unique_x_points:
            L_dist = unique_x_points[0]
            R_dist = unique_x_points[-1]
            
            # 3. 核心逻辑：扫描线配对相减，跳过凸起的截断部分
            # unique_x_points 里的点，构成了水面、水底的连续线段
            # 我们只需要确保两个点之间的中点高程是在水面之下，就把它计入宽度
            for k in range(len(unique_x_points) - 1):
                x_left = unique_x_points[k]
                x_right = unique_x_points[k+1]
                x_mid = (x_left + x_right) / 2.0
                
                # 找 x_mid 对应的实际河底高程 z_mid
                # (这里用 numpy 的插值函数快速获取中点的河底高度)
                z_mid = np.interp(x_mid, x, z)
                
                # 如果这个区间的中点在水下，说明这是一段有效水体
                if z_mid < Z:
                    net_width += (x_right - x_left)

        width_avg = None
        elevation_delta = None
        area_delta = None
        
        # 4. 正常的梯形面积计算（完美利用平移等积原理）
        if i > 0:
            prev_width = result[-1]["width(m)"]
            prev_Z = result[-1]["elevation"]
            
            width_avg = (prev_width + net_width) / 2.0
            elevation_delta = Z - prev_Z
            
            if prev_Z < lowest_elevation and cumulative_area == 0:
                area_delta = channel_area
                cumulative_area += area_delta
            elif prev_Z >= lowest_elevation:
                area_delta = width_avg * elevation_delta
                cumulative_area += area_delta
            else:
                area_delta = 0
                cumulative_area += area_delta

        result.append({
            "elevation": Z,
            "R_dist(m)": round_distance(R_dist) if R_dist is not None else None,
            "L_dist(m)": round_distance(L_dist) if L_dist is not None else None,
            "width(m)": round_width(net_width),
            "width_avg(m)": round_width(width_avg) if width_avg is not None else None,
            "elevation_delta(m)": round(elevation_delta, 2) if elevation_delta is not None else None,
            "area_delta(m^2)": round_area(area_delta) if area_delta is not None else None,
            "area(m^2)": round_area(cumulative_area),
        })

    return pd.DataFrame(result), channel_section_result