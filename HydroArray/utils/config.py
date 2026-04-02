"""
配置管理系统

提供YAML配置文件的读取、验证和管理功能
灵感来源于 neuralhydrology 的配置系统

Example:
    >>> from HydroArray.utils.config import Config
    >>> 
    >>> # 从YAML文件加载
    >>> config = Config('config.yml')
    >>> 
    >>> # 访问配置参数
    >>> print(config.model.hidden_dims)  # [64, 32, 32]
    >>> print(config.training.num_epochs)  # 20
    >>> 
    >>> # 转换为字典
    >>> config_dict = config.as_dict()
"""

import re
import random
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class Config:
    """
    配置管理类
    
    支持从YAML文件或字典加载配置，提供属性访问方式
    
    Parameters
    ----------
    config_source : Union[str, Path, Dict]
        配置文件路径或配置字典
    
    Attributes
    ----------
    model : dict
        模型相关配置
    data : dict
        数据相关配置
    training : dict
        训练相关配置
    evaluation : dict
        评估相关配置
    """
    
    # 默认配置值
    _defaults = {
        'model': {
            'type': 'convlstm',
            'input_dim': 1,
            'hidden_dims': [64, 32, 32],
            'kernel_size': 5,
            'num_layers': 3,
            'bias': True,
            'dropout': 0.0
        },
        'data': {
            'dataset': 'moving_mnist',
            'data_path': None,
            'seq_len': 10,
            'future_len': 10,
            'train_ratio': 0.9,
            'batch_size': 16,
            'num_workers': 0,
            'pin_memory': True
        },
        'training': {
            'num_epochs': 20,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'weight_decay': 0.0,
            'grad_clip': 1.0,
            'scheduler': None,
            'early_stopping': {
                'enabled': True,
                'patience': 10,
                'min_delta': 1e-4
            }
        },
        'evaluation': {
            'metrics': ['mse', 'mae'],
            'save_predictions': True,
            'visualize': True
        },
        'experiment': {
            'name': None,
            'seed': 42,
            'device': 'auto',  # 'auto', 'cuda', 'cpu'
            'save_dir': './runs',
            'log_interval': 10
        }
    }
    
    def __init__(self, config_source: Union[str, Path, Dict[str, Any]]):
        """
        初始化配置
        
        Parameters
        ----------
        config_source : Union[str, Path, Dict]
            配置文件路径或配置字典
        """
        if not YAML_AVAILABLE and not isinstance(config_source, dict):
            raise ImportError("PyYAML is required to load config files. Install with: pip install pyyaml")
        
        # 加载配置
        if isinstance(config_source, (str, Path)):
            self._cfg = self._load_from_yaml(config_source)
            self._config_path = Path(config_source)
        elif isinstance(config_source, dict):
            self._cfg = config_source.copy()
            self._config_path = None
        else:
            raise ValueError(f"config_source must be str, Path or dict, got {type(config_source)}")
        
        # 填充默认值
        self._fill_defaults()
        
        # 处理特殊字段
        self._process_config()
    
    def _load_from_yaml(self, path: Union[str, Path]) -> Dict[str, Any]:
        """从YAML文件加载配置"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        return cfg if cfg is not None else {}
    
    def _fill_defaults(self):
        """填充默认配置值"""
        for key, default_value in self._defaults.items():
            if key not in self._cfg:
                self._cfg[key] = default_value.copy() if isinstance(default_value, dict) else default_value
            elif isinstance(default_value, dict):
                # 递归填充嵌套字典
                for sub_key, sub_default in default_value.items():
                    if sub_key not in self._cfg[key]:
                        self._cfg[key][sub_key] = sub_default
    
    def _process_config(self):
        """处理特殊配置字段"""
        # 处理实验名称
        if self._cfg.get('experiment', {}).get('name') is None:
            self._cfg['experiment']['name'] = self._generate_experiment_name()
        
        # 处理路径字段 (转换为Path对象)
        path_keys = ['data_path', 'save_dir']
        for key in path_keys:
            if key in self._cfg.get('data', {}):
                if self._cfg['data'][key] is not None:
                    self._cfg['data'][key] = Path(self._cfg['data'][key])
            if key in self._cfg.get('experiment', {}):
                if self._cfg['experiment'][key] is not None:
                    self._cfg['experiment'][key] = Path(self._cfg['experiment'][key])
        
        # 处理设备设置
        device = self._cfg.get('experiment', {}).get('device', 'auto')
        if device == 'auto':
            import torch
            self._cfg['experiment']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _generate_experiment_name(self) -> str:
        """生成实验名称"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        return f"exp_{timestamp}_{random_suffix}"
    
    def __getattr__(self, name: str) -> Any:
        """属性访问方式获取配置"""
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        if name in self._cfg:
            return self._cfg[name]
        
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """字典方式获取配置"""
        return self._cfg[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持嵌套键 (如 'training.learning_rate')"""
        keys = key.split('.')
        value = self._cfg
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值，支持嵌套键"""
        keys = key.split('.')
        cfg = self._cfg
        
        for k in keys[:-1]:
            if k not in cfg:
                cfg[k] = {}
            cfg = cfg[k]
        
        cfg[keys[-1]] = value
    
    def as_dict(self) -> Dict[str, Any]:
        """返回配置字典"""
        return self._cfg.copy()
    
    def save(self, path: Union[str, Path]):
        """保存配置到YAML文件"""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required to save config files")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化的字典
        cfg_to_save = self._make_serializable(self._cfg)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg_to_save, f, default_flow_style=False, allow_unicode=True)
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可序列化的格式"""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def validate(self) -> List[str]:
        """
        验证配置有效性
        
        Returns
        -------
        List[str]
            错误信息列表，空列表表示验证通过
        """
        errors = []
        
        # 检查必需字段
        if self._cfg.get('data', {}).get('data_path') is None:
            errors.append("data.data_path is required")
        
        # 检查数值范围
        if self._cfg.get('training', {}).get('learning_rate', 0) <= 0:
            errors.append("training.learning_rate must be positive")
        
        if self._cfg.get('training', {}).get('num_epochs', 0) <= 0:
            errors.append("training.num_epochs must be positive")
        
        # 检查模型配置
        hidden_dims = self._cfg.get('model', {}).get('hidden_dims', [])
        if not isinstance(hidden_dims, list) or len(hidden_dims) == 0:
            errors.append("model.hidden_dims must be a non-empty list")
        
        return errors
    
    def is_valid(self) -> bool:
        """检查配置是否有效"""
        return len(self.validate()) == 0
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"Config(experiment='{self.experiment.name}')"
    
    def __str__(self) -> str:
        """格式化字符串"""
        lines = ["Configuration:"]
        lines.append(f"  Experiment: {self.experiment.name}")
        lines.append(f"  Model: {self.model.type}")
        lines.append(f"  Dataset: {self.data.dataset}")
        lines.append(f"  Epochs: {self.training.num_epochs}")
        lines.append(f"  Device: {self.experiment.device}")
        return "\n".join(lines)


def load_config(path: Union[str, Path]) -> Config:
    """
    从文件加载配置
    
    Parameters
    ----------
    path : Union[str, Path]
        配置文件路径
    
    Returns
    -------
    Config
        配置对象
    
    Example:
        >>> config = load_config('config.yml')
        >>> print(config.model.type)
    """
    return Config(path)


def create_default_config(save_path: Optional[Union[str, Path]] = None) -> Config:
    """
    创建默认配置
    
    Parameters
    ----------
    save_path : Optional[Union[str, Path]]
        保存路径，为None则不保存
    
    Returns
    -------
    Config
        默认配置对象
    """
    config = Config(Config._defaults)
    
    if save_path is not None:
        config.save(save_path)
    
    return config
