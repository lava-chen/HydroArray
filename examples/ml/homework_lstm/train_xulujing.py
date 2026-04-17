"""
LSTM推流预测实验 - HydroArray作业
==============================
使用HydroArray框架

实验要求:
- 预测徐六泾水文站未来48小时的流量
- 训练数据: 2023年6月1日-8月12日
- 验证数据: 2023年8月13日-8月22日
- 测试数据: 2023年8月23日-8月31日
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HydroArray.datasets import RiverDataset

train_ds = RiverDataset("./data/2023年6月-8月数据汇总_lstm.xlsx",
                        start="20230601", end="20230812")
val_ds = RiverDataset("./data/2023年6月-8月数据汇总_lstm.xlsx",
                      start="20230813", end="20230822",
                      scaler=train_ds.scaler)

model, results = train_ds.train(model="lstm", epochs=50, val_dataset=val_ds)

test_ds = RiverDataset("./data/2023年6月-8月数据汇总_lstm.xlsx",
                       start="20230823", end="20230831",
                       scaler=train_ds.scaler)
predictions, targets = test_ds.predict(model)

from HydroArray.analysis.statistics import nse, rmse
print(f"NSE: {nse(predictions, targets):.4f}")
print(f"RMSE: {rmse(predictions, targets):.2f} m3/s")
