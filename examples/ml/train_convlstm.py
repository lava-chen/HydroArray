"""
ConvLSTM Training Example

使用配置文件训练 ConvLSTM 模型。

Usage:
    python train_convlstm.py --config config_convlstm.yml
    python train_convlstm.py --config config_convlstm.yml --epochs 20
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HydroArray.ml.models.spatial.conv_lstm import train_convlstm


def main():
    parser = argparse.ArgumentParser(description='Train ConvLSTM on Moving MNIST')
    parser.add_argument('--config', type=str, default='examples/ml/config_convlstm.yml',
                        help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs from config')
    args = parser.parse_args()

    # 训练 - 使用配置文件
    model, results = train_convlstm(config=args.config)

    print(f"\nTraining complete!")
    print(f"  Best val loss: {results['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()
