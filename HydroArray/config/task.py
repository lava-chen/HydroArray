"""
任务配置

定义水文模型任务的结构和配置。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ModelType(Enum):
    """支持的模型类型"""

    # === 传统水文模型 ===
    XINANJIANG = "xinanjiang"           # 新安江模型
    HYDROMOD = "hydromod"               # HyMOD 模型
    SAC = "sac"                         # SAC-SMA 模型
    CREST = "crest"                     # CREST 模型
    SIMPLE = "simple"                   # 简单水量平衡

    # === 机器学习模型 ===
    LSTM = "lstm"                       # LSTM
    CONVLSTM = "convlstm"               # ConvLSTM
    UNET = "unet"                       # U-Net
    BILSTM = "bilstm"                   # 双向 LSTM
    DIFFUSION = "diffusion"             # 扩散模型

    # === 混合模型 ===
    PHYSICS_INFORMED = "physics_informed"  # 物理约束神经网络

    @classmethod
    def is_ml_model(cls, model_type: "ModelType") -> bool:
        """判断是否为机器学习模型"""
        ml_types = {cls.LSTM, cls.CONVLSTM, cls.UNET, cls.BILSTM, cls.DIFFUSION}
        return model_type in ml_types

    @classmethod
    def is_traditional_model(cls, model_type: "ModelType") -> bool:
        """判断是否为传统水文模型"""
        trad_types = {
            cls.XINANJIANG, cls.HYDROMOD, cls.SAC, cls.CREST, cls.SIMPLE
        }
        return model_type in trad_types


class RoutingType(Enum):
    """汇流模型类型"""
    NONE = "none"
    LINEAR = "linear"                   # 线性水库
    KINEMATIC = "kinematic"             # 运动波
    MUSKINGUM = "muskingum"             # 马斯京根


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: ModelType
    water_balance: Optional[str] = None    # 产流模型名称
    routing: Optional[RoutingType] = None  # 汇流模型
    parameters: Dict[str, Any] = field(default_factory=dict)
    routing_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_type': self.model_type.value,
            'water_balance': self.water_balance,
            'routing': self.routing.value if self.routing else None,
            'parameters': self.parameters,
            'routing_parameters': self.routing_parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """从字典创建"""
        model_type = ModelType(data['model_type'])

        routing = None
        if data.get('routing') and data['routing'] != 'none':
            routing = RoutingType(data['routing'])

        return cls(
            model_type=model_type,
            water_balance=data.get('water_balance'),
            routing=routing,
            parameters=data.get('parameters', {}),
            routing_parameters=data.get('routing_parameters', {}),
        )


@dataclass
class DataConfig:
    """数据配置"""
    data_source: str = ""                    # 数据源路径
    data_format: str = "auto"                # 数据格式 (netcdf, csv, asc, etc.)
    variables: Dict[str, str] = field(default_factory=dict)  # 变量映射
    seq_len: int = 365                       # 输入序列长度
    pred_len: int = 1                        # 预测长度

    # 时间配置
    begin_time: Optional[str] = None
    end_time: Optional[str] = None
    time_step: float = 1.0                   # 小时

    # 空间配置
    catchments: List[str] = field(default_factory=list)  # 流域列表
    bbox: Optional[List[float]] = None      # 边界框 [minx, miny, maxx, maxy]


@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "mse"

    # 早停
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.0001

    # 梯度裁剪
    grad_clip: float = 1.0

    # 验证
    validation_split: float = 0.2


@dataclass
class EvaluationConfig:
    """评估配置"""
    metrics: List[str] = field(default_factory=lambda: ["nse", "rmse"])
    save_results: bool = True
    output_dir: str = "./results"


@dataclass
class TaskConfig:
    """完整任务配置

    用于配置和管理完整的水文模拟任务。
    """
    name: str
    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    training: Optional[TrainingConfig] = None
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # 实验配置
    experiment_name: Optional[str] = None
    seed: int = 42
    device: str = "auto"
    save_dir: str = "./runs"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'name': self.name,
            'model': self.model.to_dict(),
            'data': {
                'data_source': self.data.data_source,
                'data_format': self.data.data_format,
                'variables': self.data.variables,
                'seq_len': self.data.seq_len,
                'pred_len': self.data.pred_len,
                'begin_time': self.data.begin_time,
                'end_time': self.data.end_time,
                'time_step': self.data.time_step,
                'catchments': self.data.catchments,
            },
            'evaluation': {
                'metrics': self.evaluation.metrics,
                'save_results': self.evaluation.save_results,
                'output_dir': self.evaluation.output_dir,
            },
            'experiment': {
                'seed': self.seed,
                'device': self.device,
                'save_dir': self.save_dir,
            }
        }

        if self.training:
            result['training'] = {
                'num_epochs': self.training.num_epochs,
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'optimizer': self.training.optimizer,
                'loss_function': self.training.loss_function,
                'early_stopping': self.training.early_stopping,
                'patience': self.training.patience,
                'grad_clip': self.training.grad_clip,
            }

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        """从字典创建"""
        model_config = ModelConfig.from_dict(data['model'])

        data_config = DataConfig()
        if 'data' in data:
            d = data['data']
            data_config = DataConfig(
                data_source=d.get('data_source', ''),
                data_format=d.get('data_format', 'auto'),
                variables=d.get('variables', {}),
                seq_len=d.get('seq_len', 365),
                pred_len=d.get('pred_len', 1),
                begin_time=d.get('begin_time'),
                end_time=d.get('end_time'),
                time_step=d.get('time_step', 1.0),
                catchments=d.get('catchments', []),
            )

        training_config = None
        if 'training' in data:
            t = data['training']
            training_config = TrainingConfig(
                num_epochs=t.get('num_epochs', 100),
                batch_size=t.get('batch_size', 32),
                learning_rate=t.get('learning_rate', 0.001),
                optimizer=t.get('optimizer', 'adam'),
                loss_function=t.get('loss_function', 'mse'),
                early_stopping=t.get('early_stopping', True),
                patience=t.get('patience', 10),
                grad_clip=t.get('grad_clip', 1.0),
            )

        eval_config = EvaluationConfig()
        if 'evaluation' in data:
            e = data['evaluation']
            eval_config = EvaluationConfig(
                metrics=e.get('metrics', ['nse', 'rmse']),
                save_results=e.get('save_results', True),
                output_dir=e.get('output_dir', './results'),
            )

        exp = data.get('experiment', {})

        return cls(
            name=data['name'],
            model=model_config,
            data=data_config,
            training=training_config,
            evaluation=eval_config,
            experiment_name=exp.get('experiment_name'),
            seed=exp.get('seed', 42),
            device=exp.get('device', 'auto'),
            save_dir=exp.get('save_dir', './runs'),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "TaskConfig":
        """从 YAML 文件加载"""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: str):
        """保存为 YAML 文件"""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
