"""
CAMELS-US LSTM Training Example

使用配置文件训练 CAMELS-US 数据集上的 LSTM 模型。

Usage:
    python train_camels_us.py --config config_camels_us.yml
    python train_camels_us.py --config config_camels_us.yml --epochs 100
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HydroArray.ml.models.sequence.lstm import train_lstm


def main():
    parser = argparse.ArgumentParser(description='Train LSTM on CAMELS-US')
    parser.add_argument('--config', type=str, default='config_camels_us.yml',
                        help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs from config')
    args = parser.parse_args()

    # 训练 - 使用配置文件
    model, results = train_lstm(config=args.config)

    print(f"\nTraining complete!")
    print(f"  Best val loss: {results['best_val_loss']:.6f}")
    print(f"  Test RMSE: {results['metrics']['rmse']:.4f}")
    print(f"  Test NSE: {results['metrics']['nse']:.4f}")


if __name__ == "__main__":
    main()
