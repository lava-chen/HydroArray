"""
模型训练器模块

提供统一的训练接口，支持序列模型、空间模型等多种类型
"""

import os
from typing import Optional, Dict, Any, Callable, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class BaseTrainer:
    """
    基础训练器类
    
    所有模型训练器的基类，提供通用的训练功能
    
    Example:
        >>> from HydroArray.ml.trainer import BaseTrainer
        >>> from HydroArray.ml.models import ConvLSTMCell, EncoderForecaster
        >>> 
        >>> cell = ConvLSTMCell(input_dim=1, hidden_dim=16, kernel_size=3)
        >>> model = EncoderForecaster(cell)
        >>> 
        >>> trainer = BaseTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     num_epochs=50,
        ...     learning_rate=1e-3
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        grad_clip: Optional[float] = 5.0,
        save_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            num_epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减（L2正则化）
            device: 训练设备，None则自动选择
            criterion: 损失函数，None则使用MSELoss
            optimizer: 优化器，None则使用Adam
            scheduler: 学习率调度器（可选）
            grad_clip: 梯度裁剪阈值，None则不裁剪
            save_dir: 模型保存目录
            verbose: 是否打印训练信息
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        
        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        
        # 优化器
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # 学习率调度器
        self.scheduler = scheduler
        
        # 其他设置
        self.grad_clip = grad_clip
        self.save_dir = Path(save_dir) if save_dir else Path.cwd()
        self.verbose = verbose
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
        
        if self.verbose:
            print(f"训练器初始化完成")
            print(f"  设备: {self.device}")
            print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  训练批次: {len(train_loader)}")
            if val_loader:
                print(f"  验证批次: {len(val_loader)}")
    
    def train_epoch(self) -> float:
        """训练一个epoch，返回平均损失"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = self._prepare_batch(inputs)
            targets = self._prepare_batch(targets)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.forward_pass(inputs, targets)
            loss = self.compute_loss(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """验证，返回平均损失"""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in self.val_loader:
            inputs = self._prepare_batch(inputs)
            targets = self._prepare_batch(targets)
            
            outputs = self.forward_pass(inputs, targets)
            loss = self.compute_loss(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def forward_pass(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播，子类可重写以支持特殊输入格式
        
        Args:
            inputs: 输入数据
            targets: 目标数据（用于获取预测步数等信息）
        
        Returns:
            模型输出
        """
        return self.model(inputs)
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失，子类可重写以支持自定义损失
        
        Args:
            outputs: 模型输出
            targets: 目标数据
        
        Returns:
            损失值
        """
        return self.criterion(outputs, targets)
    
    def _prepare_batch(self, batch) -> torch.Tensor:
        """准备批次数据，移动到设备"""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch
    
    def train(self) -> Dict[str, Any]:
        """
        执行完整训练流程
        
        Returns:
            训练历史记录
        """
        if self.verbose:
            print(f"\n开始训练 {self.num_epochs} 轮...")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 打印信息
            if self.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                msg = f"Epoch [{self.current_epoch}/{self.num_epochs}] "
                msg += f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e}"
                print(msg)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
        
        if self.verbose:
            print(f"\n训练完成！最佳验证损失: {self.best_val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def save_checkpoint(self, filename: str):
        """保存模型检查点"""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.save_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        torch.save(checkpoint, filepath)
        
        if self.verbose:
            print(f"  -> 保存模型: {filepath}")
    
    def load_checkpoint(self, filename: str):
        """加载模型检查点"""
        filepath = self.save_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.current_epoch = checkpoint.get('epoch', 0)
        
        if self.verbose:
            print(f"加载模型: {filepath}")
    
    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        使用训练好的模型进行预测
        
        Args:
            inputs: 输入数据
        
        Returns:
            预测结果
        """
        self.model.eval()
        inputs = inputs.to(self.device)
        return self.model(inputs)


class SequenceTrainer(BaseTrainer):
    """
    序列预测模型训练器
    
    专门用于训练序列到序列的预测模型，如 ConvLSTM、LSTM 等
    
    Example:
        >>> from HydroArray.ml.trainer import SequenceTrainer
        >>> from HydroArray.ml.models import ConvLSTMCell, EncoderForecaster
        >>> 
        >>> cell = ConvLSTMCell(input_dim=1, hidden_dim=16, kernel_size=3)
        >>> model = EncoderForecaster(cell)
        >>> 
        >>> trainer = SequenceTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     future_steps=10,
        ...     num_epochs=50
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        future_steps: int = 10,
        **kwargs
    ):
        """
        初始化序列训练器
        
        Args:
            model: 序列预测模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            future_steps: 预测步数
            **kwargs: 传递给 BaseTrainer 的其他参数
        """
        self.future_steps = future_steps
        super().__init__(model, train_loader, val_loader, **kwargs)
    
    def forward_pass(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        序列模型的前向传播
        
        自动处理需要 future_steps 参数的模型
        """
        # 检查模型是否需要 future_steps 参数
        import inspect
        sig = inspect.signature(self.model.forward)
        params = list(sig.parameters.keys())
        
        if 'future_steps' in params:
            return self.model(inputs, future_steps=self.future_steps)
        else:
            return self.model(inputs)


class SpatialTrainer(BaseTrainer):
    """
    空间预测模型训练器
    
    用于训练 U-Net 等空间预测模型
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        **kwargs
    ):
        super().__init__(model, train_loader, val_loader, **kwargs)


def create_trainer(
    model_type: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    **kwargs
) -> BaseTrainer:
    """
    工厂函数：创建对应类型的训练器
    
    Args:
        model_type: 模型类型，'sequence' | 'spatial' | 'generative'
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        **kwargs: 其他参数
    
    Returns:
        对应类型的训练器
    
    Example:
        >>> trainer = create_trainer(
        ...     model_type='sequence',
        ...     model=model,
        ...     train_loader=train_loader,
        ...     future_steps=10
        ... )
    """
    if model_type == 'sequence':
        return SequenceTrainer(model, train_loader, val_loader, **kwargs)
    elif model_type == 'spatial':
        return SpatialTrainer(model, train_loader, val_loader, **kwargs)
    else:
        return BaseTrainer(model, train_loader, val_loader, **kwargs)
