# LSTM推流预测实验 - HydroArray作业

## 实验背景

本实验是河海大学水文专业本科实验，使用LSTM神经网络预测徐六泾水文站（位于长江下游）未来48小时的流量。

## 数据说明

- **数据来源**: 2023年6月-8月徐六泾水文站观测数据
- **数据文件**: `data/2023年6月-8月数据汇总_lstm.xlsx`
- **数据列**:
  - 列0-2: 水位数据（共青圩、崇明洲头、徐六泾）
  - 列3: 流量（目标变量）
  - 列4-8: 浮标测线流速数据
  - 列9: 过水断面面积

## 数据划分

| 数据集 | 时间范围 |
|--------|----------|
| 训练集 | 2023-06-01 ~ 2023-08-12 |
| 验证集 | 2023-08-13 ~ 2023-08-22 |
| 测试集 | 2023-08-23 ~ 2023-08-31 |

## 文件结构

```
homework_lstm/
├── data/                          # 数据文件
│   └── 2023年6月-8月数据汇总_lstm.xlsx
├── outputs/                       # 输出目录
│   ├── scale_lstm/               # 归一化参数
│   ├── model/                    # 模型权重
│   ├── result/                   # 预测结果
│   └── pngs/                     # 可视化图片
├── config.yml                     # 配置文件
├── train_xulujing.py             # 训练脚本
└── homework_xulujing.ipynb       # Jupyter notebook
```

## 快速开始

### 方法一：运行Python脚本

```bash
cd examples/ml/homework_lstm
python train_xulujing.py
```

### 方法二：使用Jupyter Notebook

```bash
cd examples/ml/homework_lstm
jupyter notebook homework_xulujing.ipynb
```

## 模型架构

采用Encoder-Decoder LSTM结构：

```
Encoder (LSTM):
  - 输入: (batch, seq_len, 8)  # 8维特征
  - 隐层: 256维

Decoder (LSTM):
  - 输入: (batch, forecast_horizon, 8)
  - 隐层: 256维

输出层:
  - Linear(256 -> 64)
  - Linear(64 -> 1)
  - 输出: (batch, forecast_horizon, 1)
```

## 评分标准

1. **数据预处理** (15分)
   - 数据加载正确
   - 训练/验证/测试集划分合理

2. **模型构建** (25分)
   - LSTM层数及维度合理
   - 全连接层配置正确
   - dropout等正则化

3. **模型训练** (25分)
   - 损失函数选择正确
   - 优化器配置合理
   - 训练循环正常收敛

4. **结果可视化** (15分)
   - 预测结果对比图
   - 模型评估指标

5. **模型预测** (10分)
   - 成功预测8月23日-31日流量

6. **分析与思考** (10分)
   - 对模型性能的分析
   - 改进方向的思考

## 评估指标

- **NSE (Nash-Sutcliffe Efficiency)**: 效率系数，越接近1越好
- **RMSE (Root Mean Square Error)**: 均方根误差，越小越好
- **Corr (Correlation)**: 相关系数，越接近1越好
- **MAE (Mean Absolute Error)**: 平均绝对误差，越小越好

## 参考

- 实验要求: `AI水文实验说明-LSTM推流（2026-04-11）.pdf`
- 参考代码: 附件3中的lstm_train.py和lstm_predict.py
