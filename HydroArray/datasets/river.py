"""
River Dataset for Streamflow Forecasting

专门用于河流流量预测的数据集，支持：
- 徐六泾水文站数据格式
- Encoder-Decoder LSTM 输入格式
- 与 HydroArray.ml.train 无缝集成

Example:
    >>> from HydroArray.datasets.river import RiverDataset

    >>> # 加载徐六泾数据，5行代码完成训练
    >>> dataset = RiverDataset("data.xlsx", train_start="20230601", train_end="20230812")
    >>> model, results = dataset.train(model="lstm", epochs=50)

    >>> # 或分步操作
    >>> train_ds = RiverDataset("data.xlsx", start="20230601", end="20230812")
    >>> val_ds = RiverDataset("data.xlsx", start="20230813", end="20230822", scaler=train_ds.scaler)
    >>> from HydroArray.ml import train
    >>> model, results = train(train_ds, val_dataset=val_ds, model="lstm")
"""

import datetime
import os
from pathlib import Path
from typing import Optional, Union, Tuple, List

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def return_v_dataindex_list(value_str: str) -> List[int]:
    """返回浮标测线对应的数据索引"""
    return [int(i) + 3 for i in value_str] + [9]


class RiverDataset(Dataset):
    """
    河流水位-流量预测数据集

    专为Encoder-Decoder LSTM设计，支持徐六泾水文站等数据的格式。

    Parameters
    ----------
    excel_path : str 或 Path
        Excel数据文件路径
    start : str
        开始日期，格式YYYYMMDD
    end : str
        结束日期，格式YYYYMMDD
    seq_len : int, optional
        输入序列长度，默认48（24小时，30min间隔）
    pred_len : int, optional
        预测序列长度，默认48（24小时）
    choose_v : str, optional
        选择的浮标测线，默认'12345'
    scaler : tuple, optional
        归一化器元组(water_scaler, flow_scaler)，用于验证/测试集
    is_train : bool, optional
        是否为训练模式，默认True
    scaler_dir : str, optional
        scaler保存目录

    Attributes
    ----------
    water_scaler : MinMaxScaler
        水位数据归一化器
    flow_scaler : MinMaxScaler
        流量数据归一化器

    Example:
        >>> # 加载训练数据
        >>> train_ds = RiverDataset("data.xlsx", "20230601", "20230812")

        >>> # 加载验证数据（使用训练集的scaler）
        >>> val_ds = RiverDataset("data.xlsx", "20230813", "20230822",
        ...                       scaler=train_ds.scaler)

        >>> # 训练模型
        >>> from HydroArray.ml import train
        >>> model, results = train(train_ds, val_dataset=val_ds, model="lstm")
    """

    def __init__(
        self,
        excel_path: Union[str, Path],
        start: str,
        end: str,
        seq_len: int = 48,
        pred_len: int = 48,
        choose_v: str = '12345',
        scaler: Optional[Tuple] = None,
        is_train: bool = True,
        scaler_dir: Optional[str] = None,
    ):
        self.excel_path = Path(excel_path)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.choose_v = choose_v
        self.is_train = is_train

        # 读取数据
        all_data = pd.read_excel(excel_path, index_col=0).interpolate(
            method='linear', limit_direction='both'
        ).astype('float32')

        # 调整日期范围
        start_dt = datetime.datetime.strptime(start, '%Y%m%d')
        end_dt = datetime.datetime.strptime(end, '%Y%m%d')

        if is_train:
            start_dt = start_dt - datetime.timedelta(days=1)
            end_dt = end_dt + datetime.timedelta(days=pred_len // 48)
        else:
            start_dt = start_dt - datetime.timedelta(days=1 + seq_len // 48) + datetime.timedelta(minutes=30)
            end_dt = end_dt + datetime.timedelta(days=pred_len // 48) - datetime.timedelta(minutes=30)

        # 提取数据（只取存在的日期）
        data_index = pd.date_range(start_dt, end_dt, freq='30min')
        # 只保留存在于数据的日期
        valid_dates = data_index.intersection(all_data.index)
        selected_data = all_data.loc[valid_dates].values

        # 构建样本
        X_encoder1, X_encoder2, X_decoder1, X_decoder2, y_data = [], [], [], [], []
        temp_index = return_v_dataindex_list(choose_v)

        for i in range(len(selected_data) - seq_len - pred_len + 1):
            temp_ex = selected_data[i:i + seq_len, 0:3]  # 水位
            tem_ex2 = selected_data[i:i + seq_len, temp_index]  # 浮标流速
            temp_dx = selected_data[i + seq_len:i + seq_len + pred_len, 0:3]
            tem_dx2 = selected_data[i + seq_len:i + seq_len + pred_len, temp_index]
            temp_y = selected_data[i + seq_len:i + seq_len + pred_len, 3]  # 流量

            # 检查NaN
            if (np.isnan(temp_ex).any() or np.isnan(tem_ex2).any() or
                np.isnan(temp_dx).any() or np.isnan(tem_dx2).any() or np.isnan(temp_y).any()):
                continue

            X_encoder1.append(temp_ex)
            X_encoder2.append(tem_ex2)
            X_decoder1.append(temp_dx)
            X_decoder2.append(tem_dx2)
            y_data.append(temp_y)

        self.begin_i = start
        self.end_i = end

        # 合并特征
        X_encoder = np.concatenate([np.array(X_encoder1), np.array(X_encoder2)], axis=2)
        X_decoder = np.concatenate([np.array(X_decoder1), np.array(X_decoder2)], axis=2)

        # 归一化
        from sklearn.preprocessing import MinMaxScaler
        self.water_scaler = MinMaxScaler()
        self.flow_scaler = MinMaxScaler()

        e_data = X_encoder.reshape(-1, 4 + len(choose_v))
        d_data = X_decoder.reshape(-1, 4 + len(choose_v))

        self.water_e = self.water_scaler.fit_transform(e_data).reshape(-1, seq_len, 4 + len(choose_v))
        self.water_d = self.water_scaler.transform(d_data).reshape(-1, pred_len, 4 + len(choose_v))
        self.flow = self.flow_scaler.fit_transform(np.array(y_data).reshape(-1, 1)).reshape(-1, pred_len)

        # 保存/加载scaler
        scaler_dir = scaler_dir or "./outputs/scale_lstm"
        if is_train:
            os.makedirs(scaler_dir, exist_ok=True)
            joblib.dump(self.water_scaler, f'{scaler_dir}/water_scaler.pkl')
            joblib.dump(self.flow_scaler, f'{scaler_dir}/flow_scaler.pkl')
        elif scaler:
            self.water_scaler, self.flow_scaler = scaler
        else:
            self.water_scaler = joblib.load(f'{scaler_dir}/water_scaler.pkl')
            self.flow_scaler = joblib.load(f'{scaler_dir}/flow_scaler.pkl')

        self.scaler = (self.water_scaler, self.flow_scaler)

        # 输入维度
        self.input_dim = 4 + len(choose_v)

    def __len__(self):
        return len(self.water_e)

    def __getitem__(self, idx):
        x_e = torch.tensor(self.water_e[idx], dtype=torch.float32)
        x_d = torch.tensor(self.water_d[idx], dtype=torch.float32)
        y = torch.tensor(self.flow[idx], dtype=torch.float32)
        # 返回(x_e, x_d)作为输入，y作为目标
        return (x_e, x_d), y

    def train(
        self,
        model: str = "lstm",
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        hidden_dim: int = 64,
        num_layers: int = 2,
        val_dataset: Optional['RiverDataset'] = None,
        save_dir: Optional[str] = None,
        **kwargs
    ):
        """
        在此数据集上训练模型（需要Encoder-Decoder LSTM）

        Parameters
        ----------
        model : str
            模型类型，当前仅支持'lstm'
        epochs : int
            训练轮数
        batch_size : int
            批大小
        learning_rate : float
            学习率
        hidden_dim : int
            LSTM隐藏层维度
        num_layers : int
            LSTM层数
        val_dataset : RiverDataset, optional
            验证数据集
        save_dir : str, optional
            结果保存目录

        Returns
        -------
        Tuple
            (model, results)

        Example:
            >>> dataset = RiverDataset("data.xlsx", "20230601", "20230812")
            >>> model, results = dataset.train(model="lstm", epochs=50)
        """
        # Seq2SeqLSTM 在当前模块定义
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建模型
        model = Seq2SeqLSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            output_size=1,
            forecast_horizon=self.pred_len
        ).to(device)

        # DataLoader
        train_loader = DataLoader(self, batch_size=batch_size, shuffle=True)

        if val_dataset:
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            n = len(self)
            n_val = int(n * 0.1)
            indices = list(range(n))
            np.random.shuffle(indices)
            val_idx = set(indices[:n_val])
            train_idx = indices[n_val:]
            val_loader = DataLoader(
                self, batch_size=batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(list(val_idx))
            )

        # 训练
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

        train_losses, val_losses = [], []
        best_val_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for (x_e, x_d), y in train_loader:
                x_e, x_d, y = x_e.to(device), x_d.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x_e, x_d)
                loss = criterion(pred.squeeze(-1), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for (x_e, x_d), y in val_loader:
                    x_e, x_d, y = x_e.to(device), x_d.to(device), y.to(device)
                    pred = model(x_e, x_d)
                    val_loss += criterion(pred.squeeze(-1), y).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        model.load_state_dict(best_state)

        return model, {'train_losses': train_losses, 'val_losses': val_losses}

    def predict(self, model: 'torch.nn.Module') -> Tuple[np.ndarray, np.ndarray]:
        """
        使用模型进行预测

        Parameters
        ----------
        model : nn.Module
            训练好的模型

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (predictions, targets) - 预测值和真实值（反归一化后）
        """
        device = next(model.parameters()).device
        model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for (x_e, x_d), y in self:
                x_e, x_d = x_e.to(device), x_d.to(device)
                pred = model(x_e, x_d)

                # 确保是numpy
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                if isinstance(y, torch.Tensor):
                    y = y.cpu().numpy()

                pred = self.flow_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1, self.pred_len)
                target = self.flow_scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1, self.pred_len)
                all_preds.append(pred)
                all_targets.append(target)

        return np.concatenate(all_preds), np.concatenate(all_targets)


class Seq2SeqLSTM(torch.nn.Module):
    """
    Encoder-Decoder LSTM for Sequence-to-Sequence Prediction

    专为河道流量预测设计的Seq2Seq模型。

    Parameters
    ----------
    input_size : int
        输入特征维度
    hidden_size : int
        LSTM隐藏层维度
    output_size : int
        输出维度
    forecast_horizon : int
        预测步数
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, forecast_horizon: int = 48):
        super().__init__()
        self.forecast_horizon = forecast_horizon

        self.encoder_lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.decoder_lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size, 64)
        self.fc2 = torch.nn.Linear(64, output_size)

    def forward(self, encoder_input, decoder_input):
        _, (hidden, cell) = self.encoder_lstm(encoder_input)
        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        output = self.fc1(decoder_output)
        output = self.fc2(output)
        return output
