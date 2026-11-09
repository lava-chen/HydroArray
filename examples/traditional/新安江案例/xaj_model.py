"""
新安江模型案例 · 模型运行核心
============================
**关键修复**: 不再使用 states["runoff_generated"] (R_total, 产流量),
而是使用 RS + RI + RG (三水源出流) + CI/CG 退水 + Muskingum 汇流,
得到流域出口径流深 (mm/步)。

新安江三水源模型正确流程:
  1. water_balance() → RS, RI, RG (三水源, 已扣自由水蓄水库)
  2. RI 经线性水库退水 (CI): QI[t] = CI*QI[t-1] + (1-CI)*RI[t]
  3. RG 经线性水库退水 (CG): QG[t] = CG*QG[t-1] + (1-CG)*RG[t]
  4. 河网总入流 = RS + QI + QG
  5. Muskingum 汇流 → 出口径流深 (mm/步)

注意: RS+RI+RG ≠ R_total, 因为自由水蓄水库会留一部分 S 给下时段。
"""
from __future__ import annotations
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import HydroArray as ha


class XAJRunner:
    """新安江模型运行器: 封装产流 + 三水源退水 + Muskingum 汇流。

    与旧版 run_xaj() 的区别:
    - 旧版: 取 states["runoff_generated"] (R_total) 直接做 Muskingum → 错误
    - 新版: 取 RS+RI+RG, 先做 CI/CG 线性水库退水, 再做 Muskingum → 正确
    """

    def __init__(self, params: ha.XinAnjiangParameters, area_km2: float,
                 apply_lag: bool = False):
        self.params = params
        self.area_km2 = area_km2
        self.apply_lag = apply_lag
        self._subbasins = self._load_subbasins()
        self._area_weights = np.array([r["area_ratio"] for r in self._subbasins], dtype=float)
        self._muskingum_n_daily = np.array([r["muskingum_n_daily"] for r in self._subbasins], dtype=int)
        self._muskingum_n_event = np.array([r["muskingum_n_event"] for r in self._subbasins], dtype=int)
        self._reset()

    @staticmethod
    def _load_subbasins() -> list[dict]:
        path = Path(__file__).with_name("subbasins.yaml")
        if not path.exists():
            return [{"area_ratio": 1.0, "muskingum_n_daily": 0, "muskingum_n_event": 0}]
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)["subbasins"]

    def _reset(self):
        """重置模型与内部状态 (每次 run 前调用)。"""
        p = self.params
        self.model = ha.XinAnjiangModel()
        nodes = [ha.GridNode(node_id=0, x=0.0, y=0.0, area=self.area_km2)]
        self.model.initialize(nodes, p)
        self.routing = ha.MuskingumRouting()
        self.routing.initialize(nodes, ha.MuskingumParameters(K=p.KE, x=p.xe))
        # 退水水库状态 (标量, 单节点)
        self._QI_prev = 0.0   # 壤中流退水库蓄量 (mm)
        self._QG_prev = 0.0   # 地下水退水库蓄量 (mm)
        self._lag_history = []

    def _initialize_discharge(self, q0_mm: float):
        """用场次首时刻流量初始化慢水与河道状态。"""
        q0 = max(float(q0_mm), 0.0)
        self._QG_prev = q0
        self.routing.set_states({
            "Q_in_history": np.array([[q0]], dtype=float),
            "Q_out_history": np.array([[q0]], dtype=float),
        })

    def _initialize_soil_from_discharge(self, q0_mm: float, p0_mm: float = 0.0):
        """根据初始实测流量反推土壤含水量。

        关键: 模型默认土壤是干的, 起算时大量降雨被吸收, 产流偏少。
        实际次洪模拟时, 流域已经处于"前期湿润"状态, 应初始化为接近饱和。

        策略:
          - 退水水库按"基流主导"设置: QG = 0.7×Q0, QI = 0.15×Q0
          - 土壤含水量按 q0 反推湿润度, 但不能太高 (避免前几小时小雨就大量产流)
          - w_ratio 范围 0.50-0.80
        """
        q0 = max(float(q0_mm), 0.0)
        # ── 1) 退水水库初始化 ──
        self._QG_prev = q0 * 0.7
        self._QI_prev = q0 * 0.15

        # ── 2) 河道状态初始化 ──
        self.routing.set_states({
            "Q_in_history": np.array([[q0]], dtype=float),
            "Q_out_history": np.array([[q0]], dtype=float),
        })

        # ── 3) 土壤含水量初始化 ──
        WM = float(self.params.WM)
        if q0 > 20.0:
            w_ratio = 0.80
        elif q0 > 10.0:
            w_ratio = 0.70
        elif q0 > 5.0:
            w_ratio = 0.60
        elif q0 > 2.0:
            w_ratio = 0.50
        else:
            w_ratio = 0.40
        # 三层分配: 偏下层 (深层含水更稳定)
        wu = min(float(self.params.WUM), w_ratio * WM * 0.12)
        wl = min(float(self.params.WLM), w_ratio * WM * 0.30)
        wd = max(0.0, w_ratio * WM - wu - wl)
        # 自由水蓄水库: 30% 蓄满
        SM = float(self.params.SM)
        s_init = SM * 0.30
        fr_init = 0.4 if q0 > 5.0 else 0.2

        self.model.set_states({
            "WU": np.array([wu]),
            "WL": np.array([wl]),
            "WD": np.array([wd]),
            "S": np.array([s_init]),
            "FR": np.array([fr_init]),
        })

    def _route_with_lag(self, q_routed: float) -> float:
        """按 L 个时段施加可连续率定的线性插值滞时。"""
        self._lag_history.append(max(float(q_routed), 0.0))
        lag = max(float(self.params.L), 0.0)
        lower = int(np.floor(lag))
        frac = lag - lower

        def _history(offset: int) -> float:
            idx = len(self._lag_history) - 1 - offset
            return self._lag_history[idx] if idx >= 0 else self._lag_history[0]

        q_lower = _history(lower)
        q_upper = _history(lower + 1)
        return (1.0 - frac) * q_lower + frac * q_upper

    def _route_muskingum_segments(self, q_in: np.ndarray, prev_in: np.ndarray,
                                  prev_out: np.ndarray, segments: np.ndarray,
                                  step_hours: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """逐子流域做 N_i 段 Muskingum 演算, 返回各子流域出口贡献。"""
        p = self.params
        k_m = float(p.KE)
        x_m = float(p.xe)
        dt = float(step_hours)
        denom = k_m * (1.0 - x_m) + 0.5 * dt
        c0 = (0.5 * dt - k_m * x_m) / denom
        c1 = (0.5 * dt + k_m * x_m) / denom
        c2 = (k_m * (1.0 - x_m) - 0.5 * dt) / denom

        q_out = np.asarray(q_in, dtype=float).copy()
        for i, n_seg in enumerate(segments):
            q = q_out[i]
            for j in range(int(n_seg)):
                out = c0 * q + c1 * prev_in[i, j] + c2 * prev_out[i, j]
                prev_in[i, j] = q
                prev_out[i, j] = max(out, 0.0)
                q = prev_out[i, j]
            q_out[i] = q
        return q_out, prev_in, prev_out

    def _run_semidistributed(self, precip, pet, step_hours: float,
                             initial_outflow_mm: float | None = None,
                             init_soil: bool = True) -> np.ndarray:
        """10 子流域产流 + 原始段数表控制的汇流结构。"""
        precip = np.asarray(precip, dtype=float)
        if precip.ndim != 2:
            raise ValueError("半分布式运行需要二维降雨输入 (n_steps, n_subbasins)")
        n_steps, n_nodes = precip.shape
        if n_nodes != len(self._area_weights):
            raise ValueError(f"降雨子流域数 {n_nodes} 与 subbasins.yaml 不一致")

        pet = np.asarray(pet, dtype=float)
        if pet.ndim == 1:
            pet_2d = np.repeat(pet[:, None], n_nodes, axis=1)
        else:
            pet_2d = pet

        p = self.params
        nodes = [
            ha.GridNode(node_id=i, x=0.0, y=0.0, area=float(self.area_km2 * self._area_weights[i]))
            for i in range(n_nodes)
        ]
        model = ha.XinAnjiangModel()
        model.initialize(nodes, p)

        q0 = max(float(initial_outflow_mm or 0.0), 0.0)
        qi_prev = np.zeros(n_nodes, dtype=float)
        qg_prev = np.full(n_nodes, q0, dtype=float)
        if init_soil and initial_outflow_mm is not None:
            wm = float(p.WM)
            if q0 > 20.0:    w_ratio = 0.80
            elif q0 > 10.0:  w_ratio = 0.70
            elif q0 > 5.0:   w_ratio = 0.60
            elif q0 > 2.0:   w_ratio = 0.50
            else:            w_ratio = 0.40
            wu = np.full(n_nodes, min(float(p.WUM), w_ratio * wm * 0.12))
            wl = np.full(n_nodes, min(float(p.WLM), w_ratio * wm * 0.30))
            wd = np.full(n_nodes, max(0.0, w_ratio * wm - wu[0] - wl[0]))
            model.set_states({
                "WU": wu, "WL": wl, "WD": wd,
                "S": np.full(n_nodes, float(p.SM) * 0.30),
                "FR": np.full(n_nodes, 0.4 if q0 > 5.0 else 0.2),
            })

        segments = self._muskingum_n_event if step_hours <= 1.5 else self._muskingum_n_daily
        max_seg = int(max(segments.max(initial=0), 1))
        prev_in = np.full((n_nodes, max_seg), q0 * self._area_weights[:, None], dtype=float)
        prev_out = prev_in.copy()
        cs_prev = q0
        q_sim = np.empty(n_steps, dtype=float)
        self._lag_history = []

        for t in range(n_steps):
            rs, ri, rg, _ = model.water_balance(
                step_hours,
                np.maximum(precip[t], 0.0),
                np.maximum(pet_2d[t], 0.0),
            )
            qi = float(p.CI) * qi_prev + (1.0 - float(p.CI)) * ri
            qg = float(p.CG) * qg_prev + (1.0 - float(p.CG)) * rg
            qi_prev = qi
            qg_prev = qg
            local_depth = np.maximum(rs + qi + qg, 0.0)

            if segments.max(initial=0) <= 0:
                total = float(np.sum(local_depth * self._area_weights))
                routed = float(p.CS) * cs_prev + (1.0 - float(p.CS)) * total
                cs_prev = routed
            else:
                weighted = local_depth * self._area_weights
                routed_sub, prev_in, prev_out = self._route_muskingum_segments(
                    weighted, prev_in, prev_out, segments, step_hours
                )
                routed = float(np.sum(routed_sub))
            q_sim[t] = self._route_with_lag(routed) if self.apply_lag else routed
        return np.maximum(q_sim, 0.0)

    def _step(self, step_hours: float, p_val: float, e_val: float) -> float:
        """单步推进: 产流 → 三水源退水 → Muskingum 汇流。

        Returns:
            出口径流深 (mm/步, 已保证非负)
        """
        p = self.params
        P = np.array([max(float(p_val), 0.0)])
        E = np.array([max(float(e_val), 0.0)])

        RS, RI, RG, _ = self.model.water_balance(step_hours, P, E)

        rs = float(RS[0])
        ri = float(RI[0])
        rg = float(RG[0])

        # ── 三水源退水 (线性水库) ──
        # QI[t] = CI * QI[t-1] + (1-CI) * RI[t]
        # QG[t] = CG * QG[t-1] + (1-CG) * RG[t]
        # 注: CI/CG 是"消退系数", 值越大退水越慢
        ci = float(p.CI)
        cg = float(p.CG)
        qi = ci * self._QI_prev + (1.0 - ci) * ri
        qg = cg * self._QG_prev + (1.0 - cg) * rg
        self._QI_prev = qi
        self._QG_prev = qg

        # ── 河网总入流 = 地表径流 + 退水后壤中流 + 退水后地下水 ──
        total_inflow = np.array([max(rs + qi + qg, 0.0)])

        # ── Muskingum 河道汇流 ──
        q_out = float(np.maximum(self.routing.route(step_hours, total_inflow)[0], 0.0))
        return self._route_with_lag(q_out) if self.apply_lag else q_out

    # ─────────────────────── 向量化快速版 (单节点) ───────────────────────
    # 内联所有计算到单个 Python 循环, 避免每步:
    #   - model.water_balance() / routing.route() 函数调用
    #   - np.asarray / np.broadcast_to / np.zeros 数组分配
    #   - _route_with_lag 的 list 操作
    # 标量算术用 Python float, 与原版位级一致 (操作顺序与 numpy 单元素等价)
    #
    # 使用方法: 用 _init_fast_state() 初始化一次状态, 之后 set_params_fast()
    # 更新参数, run_fast() 复用状态跑新数据, 避免重复创建模型对象。
    _FAST_STATE: dict | None = None  # 单节点标量状态 (跨调用复用)

    def _init_fast_state(self):
        """初始化单节点标量状态, 跨调用复用。"""
        self._FAST_STATE = {
            "wu": 0.0, "wl": 0.0, "wd": 0.0,
            "S": 0.0, "FR": 1.0,
            "QI": 0.0, "QG": 0.0,
            "Q_in_prev": 0.0, "Q_out_prev": 0.0,
        }

    def set_params_fast(self, p, area_km2: float | None = None,
                        apply_lag: bool = False,
                        initial_q_mm: float = 0.0,
                        init_soil: bool = True,
                        q0_for_soil: float = 0.0):
        """设置参数 + 初始化快速状态。"""
        if area_km2 is not None:
            self.area_km2 = float(area_km2)
        self.apply_lag = bool(apply_lag)
        if self._FAST_STATE is None:
            self._init_fast_state()
        s = self._FAST_STATE

        # Muskingum 初值
        q0 = max(float(initial_q_mm), 0.0)
        s["Q_in_prev"] = q0
        s["Q_out_prev"] = q0

        if init_soil and q0_for_soil > 0.0:
            q0s = max(float(q0_for_soil), 0.0)
            s["QG"] = q0s * 0.7
            s["QI"] = q0s * 0.15
            WM = max(float(p.WM), 1.0)
            if q0s > 20.0:    w_ratio = 0.80
            elif q0s > 10.0:  w_ratio = 0.70
            elif q0s > 5.0:   w_ratio = 0.60
            elif q0s > 2.0:   w_ratio = 0.50
            else:             w_ratio = 0.40
            s["wu"] = min(float(p.WUM), w_ratio * WM * 0.12)
            s["wl"] = min(float(p.WLM), w_ratio * WM * 0.30)
            s["wd"] = max(0.0, w_ratio * WM - s["wu"] - s["wl"])
            SM = max(float(p.SM), 1e-6)
            s["S"] = SM * 0.30
            s["FR"] = 0.4 if q0s > 5.0 else 0.2
        else:
            s["QG"] = q0
            s["QI"] = 0.0
            WUM = float(p.WUM); WLM = float(p.WLM); WDM = float(p.WDM)
            WM = max(WUM + WLM + WDM, 1.0)
            s["wu"] = WUM * 0.5
            s["wl"] = WLM * 0.5
            s["wd"] = WDM * 0.5
            SM = max(float(p.SM), 1e-6)
            s["S"] = SM * 0.5
            s["FR"] = 1.0

    def run_fast(self, precip, pet, step_hours: float) -> np.ndarray:
        """向量化快速版 run(): 单节点标量循环, 与原版结果位级一致。"""
        p = self.params
        if self._FAST_STATE is None:
            self._init_fast_state()
        s = self._FAST_STATE

        # ── 预计算常数 ──
        B  = float(p.B);  EX = float(p.EX);  IMP = float(p.IMP)
        K_et = float(p.K); C_et = float(p.C)
        WUM = float(p.WUM); WLM = float(p.WLM); WDM = float(p.WDM)
        WM = max(WUM + WLM + WDM, 1.0)
        WMM = WM * (1.0 + B)
        SM = max(float(p.SM), 0.0)
        SMM = SM * (1.0 + EX)
        ki = max(float(p.KI), 0.0)
        kg = max(float(p.KG), 0.0)
        if ki + kg > 0.7:
            scale = 0.7 / (ki + kg)
            ki *= scale; kg *= scale
        ci = float(p.CI); cg = float(p.CG)

        # Muskingum 系数
        K_m = float(p.KE); x_m = float(p.xe)
        dt = float(step_hours)
        denom = K_m * (1.0 - x_m) + 0.5 * dt
        C0 = (0.5 * dt - K_m * x_m) / denom
        C1 = (0.5 * dt + K_m * x_m) / denom
        C2 = (K_m * (1.0 - x_m) - 0.5 * dt) / denom

        # Lag 预计算
        lag = max(float(p.L), 0.0)
        lag_lo = int(lag) if lag == int(lag) else int(np.floor(lag))
        lag_frac = lag - lag_lo
        inv_frac = 1.0 - lag_frac

        # 当前状态
        wu = s["wu"]; wl = s["wl"]; wd = s["wd"]
        S_ = s["S"]; FR = s["FR"]
        QI = s["QI"]; QG = s["QG"]
        Q_in_prev = s["Q_in_prev"]; Q_out_prev = s["Q_out_prev"]

        # Lag 缓冲: 长度 = lag_lo + 2 (满足 lag_lo+1 索引)
        if self.apply_lag:
            lag_buf = [0.0] * (lag_lo + 2)
        else:
            lag_buf = None

        n = len(precip)
        q_sim = np.empty(n, dtype=float)
        one_minus_B = 1.0 / (1.0 + B)

        for t in range(n):
            # ── 1) 三层蒸散发 ──
            p_t = precip[t] if precip[t] > 0.0 else 0.0
            ep_t = (pet[t] if pet[t] > 0.0 else 0.0) * K_et
            w_before = wu + wl + wd

            if ep_t <= 0.0:
                eu, el, ed = 0.0, 0.0, 0.0
            else:
                eu = ep_t if ep_t < wu + p_t else wu + p_t
                remain = ep_t - eu
                if remain <= 0.0:
                    el, ed = 0.0, 0.0
                else:
                    if WLM > 0.0 and wl >= C_et * WLM:
                        el = remain * (wl / WLM)
                        if el > wl: el = wl
                        ed = 0.0
                    elif WLM > 0.0 and C_et * remain <= wl:
                        el = C_et * remain
                        if el > wl: el = wl
                        ed = 0.0
                    else:
                        el = wl
                        ed = C_et * remain - el
                        if ed < 0.0: ed = 0.0

            E_actual = eu + el + ed
            PE = p_t - E_actual

            # ── 2) 产流 + 三水源 ──
            if PE <= 0.0:
                deficit = p_t
                if eu > deficit: eu -= deficit; deficit = 0.0
                else: deficit -= eu
                if el > deficit: el -= deficit; deficit = 0.0
                else: deficit -= el
                ed2 = ed - deficit
                if ed2 < 0.0: ed2 = 0.0
                wu -= eu
                wl -= el
                wd -= ed2
                if wd < 0.0: wd = 0.0
                R_pervious = 0.0
            else:
                if WM > 0.0:
                    w_ratio = w_before / WM
                    if w_ratio < 0.0: w_ratio = 0.0
                    elif w_ratio > 1.0: w_ratio = 1.0
                    a = WMM * (1.0 - (1.0 - w_ratio) ** one_minus_B)
                else:
                    a = 0.0
                if a + PE <= WMM:
                    R_pervious = (PE + w_before - WM
                                  + WM * (1.0 - (PE + a) / WMM) ** (1.0 + B))
                else:
                    R_pervious = PE - (WM - w_before)
                if R_pervious < 0.0: R_pervious = 0.0
                elif R_pervious > PE: R_pervious = PE

                # fill_tension_storage(PE - R_pervious, wu, wl, wd, WUM, WLM, WDM)
                surplus = PE - R_pervious
                add = WUM - wu
                if add < 0.0: add = 0.0
                if add > surplus: add = surplus
                wu += add; surplus -= add
                add = WLM - wl
                if add < 0.0: add = 0.0
                if add > surplus: add = surplus
                wl += add; surplus -= add
                add = WDM - wd
                if add < 0.0: add = 0.0
                if add > surplus: add = surplus
                wd += add

            R_i = R_pervious * (1.0 - IMP) + p_t * IMP

            # ── 3) 自由水蓄水库分水源 ──
            if SM > 0.0:
                if S_ < 0.0: S_ = 0.0
                elif S_ > SM: S_ = SM
                available = S_ + R_i
                if available - SM > 0.0:
                    RS = available - SM
                else:
                    RS = 0.0
                free_storage = available if available < SM else SM
                RI = ki * free_storage
                RG = kg * free_storage
                S_new = free_storage - RI - RG
                if S_new < 0.0: S_new = 0.0
                elif S_new > SM: S_new = SM
                S_ = S_new
                ratio = 1.0 - S_ / SMM if SMM > 0.0 else 0.0
                if ratio < 0.0: ratio = 0.0
                FR = (1.0 - ratio ** EX) if EX != 0 else 1.0
                if FR < 1e-6: FR = 1e-6
                elif FR > 1.0: FR = 1.0
            else:
                RS = R_i; S_ = 0.0; FR = 1.0; RI = 0.0; RG = 0.0

            # ── 4) 退水 (线性水库) ──
            qi = ci * QI + (1.0 - ci) * RI
            qg = cg * QG + (1.0 - cg) * RG

            # ── 5) 河网总入流 + Muskingum ──
            total = RS + qi + qg
            if total < 0.0: total = 0.0
            q_out = C0 * total + C1 * Q_in_prev + C2 * Q_out_prev
            if q_out < 0.0: q_out = 0.0

            # ── 6) Lag ──
            if lag_buf is not None:
                # 推入末尾
                for j in range(len(lag_buf) - 1):
                    lag_buf[j] = lag_buf[j + 1]
                lag_buf[-1] = q_out
                # 计算 lag_lo, lag_lo+1 历史
                idx_lo = len(lag_buf) - 1 - lag_lo
                idx_hi = idx_lo - 1
                if idx_lo < 0: idx_lo = 0; q_lo = lag_buf[0]
                else: q_lo = lag_buf[idx_lo]
                if idx_hi < 0: q_hi = lag_buf[0]
                else: q_hi = lag_buf[idx_hi]
                q_sim[t] = inv_frac * q_lo + lag_frac * q_hi
            else:
                q_sim[t] = q_out

            # ── 7) 更新状态 ──
            wu, wl, wd = wu, wl, wd
            QI = qi; QG = qg
            Q_in_prev = total; Q_out_prev = q_out

        # 保存状态供下次复用
        s["wu"] = wu; s["wl"] = wl; s["wd"] = wd
        s["S"] = S_; s["FR"] = FR
        s["QI"] = QI; s["QG"] = QG
        s["Q_in_prev"] = Q_in_prev; s["Q_out_prev"] = Q_out_prev
        return q_sim

    def run(self, precip, pet, step_hours: float,
            initial_outflow_mm: float | None = None,
            init_soil: bool = True,
            warmup_h: int = 0,
            warmup_no_rain: bool = False) -> np.ndarray:
        """连续运行 n 步, 返回出口径流深序列 (mm/步)。

        内部优先使用向量化 run_fast() (单节点 5-30x 加速),
        状态在 _FAST_STATE 复用, 避免每步 numpy 分配与函数调用。

        Args:
            precip: 面雨量序列 (mm/步), array-like
            pet: 面蒸发序列 (mm/步), array-like
            step_hours: 步长 (h), 日模型 24, 次洪 1
            initial_outflow_mm: 场次首时刻实测流量 (mm/h), 用于初始化
                - 默认 init_soil=True: 反推土壤含水量
                - 日模型下应传 None (从 0 状态起算)
            warmup_h: 保留参数, 暂未启用 (设为 0 即与原始行为一致)
            warmup_no_rain: 保留参数, 暂未启用

        Returns:
            q_sim: 出口径流深 (mm/步), ndarray
        """
        precip_arr = np.asarray(precip)
        if precip_arr.ndim == 2:
            return self._run_semidistributed(
                precip_arr, pet, step_hours,
                initial_outflow_mm=initial_outflow_mm,
                init_soil=init_soil,
            )

        # 单节点 case 直接走快速版
        if len(self.model._nodes) == 1 and not warmup_h:
            if initial_outflow_mm is not None:
                self.set_params_fast(
                    self.params, self.area_km2, self.apply_lag,
                    initial_q_mm=initial_outflow_mm,
                    init_soil=init_soil,
                    q0_for_soil=initial_outflow_mm if init_soil else 0.0,
                )
            else:
                self.set_params_fast(
                    self.params, self.area_km2, self.apply_lag,
                    initial_q_mm=0.0, init_soil=False,
                )
            return self.run_fast(precip, pet, step_hours)

        # 多节点 / 兼容旧代码路径
        self._reset()
        if initial_outflow_mm is not None:
            if init_soil:
                self._initialize_soil_from_discharge(initial_outflow_mm)
            else:
                self._initialize_discharge(initial_outflow_mm)
        n = len(precip)
        q_sim = np.empty(n, dtype=float)
        for t in range(n):
            q_sim[t] = self._step(step_hours, float(precip[t]), float(pet[t]))
        return q_sim

    def get_states(self) -> dict:
        """获取当前模型状态 (可用于跨尺度传递)。"""
        return {
            "xaj_states": self.model.get_states(),
            "routing_states": self.routing.get_states(),
            "QI_prev": self._QI_prev,
            "QG_prev": self._QG_prev,
        }

    def set_states(self, states: dict):
        """设置模型状态 (从日模型传递到次洪模型)。"""
        if "xaj_states" in states:
            self.model.set_states(states["xaj_states"])
        if "routing_states" in states:
            self.routing.set_states(states["routing_states"])
        self._QI_prev = states.get("QI_prev", 0.0)
        self._QG_prev = states.get("QG_prev", 0.0)


def apply_param_dict(params: ha.XinAnjiangParameters,
                     updates: dict) -> ha.XinAnjiangParameters:
    """将参数字典应用到 XinAnjiangParameters, 维护物理一致性。

    - WM = WUM + WLM + WDM
    - KI + KG <= 0.7 (新安江自由水蓄水库出流比例约束)
    """
    for k, v in updates.items():
        if hasattr(params, k):
            setattr(params, k, float(v))

    # 三层张力水一致性
    wdm_min = 5.0
    if params.WUM + params.WLM > params.WM - wdm_min:
        scale = (params.WM - wdm_min) / (params.WUM + params.WLM)
        params.WUM *= scale
        params.WLM *= scale
    params.WDM = params.WM - params.WUM - params.WLM

    # KI + KG 约束
    if params.KI + params.KG > 0.7:
        scale = 0.7 / (params.KI + params.KG)
        params.KI *= scale
        params.KG *= scale

    return params


def clone_params(params: ha.XinAnjiangParameters) -> ha.XinAnjiangParameters:
    """深拷贝参数对象。"""
    return ha.XinAnjiangParameters(
        **{k: getattr(params, k) for k in params.__dataclass_fields__}
    )
