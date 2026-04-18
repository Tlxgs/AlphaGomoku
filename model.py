
import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock(nn.Module):
    """残差块 - 预激活设计"""
    def __init__(self, channels, dropout_rate=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        # 预激活：BN -> ReLU -> Conv
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return residual + out


class PolicyValueNet(nn.Module):
    """策略价值网络 - 增强版（加深网络 + 增强价值头）"""

    def __init__(self, board_size=15,
                 num_res_blocks=10,
                 channels=128,
                 policy_conv_out=4,
                 policy_hidden=450,
                 value_channels=64,
                 value_hidden1=128,
                 value_hidden2=32,
                 dropout_rate=0.1,
                 input_channels=2):

        super().__init__()
        self.board_size = board_size
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.dropout_rate = dropout_rate
        self.input_channels = input_channels

        # 输入卷积层
        self.conv_input = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)

        # 残差块 (现在会根据num_res_blocks生成更深的网络)
        self.res_blocks = nn.ModuleList([
            ResBlock(channels, dropout_rate)
            for _ in range(num_res_blocks)
        ])

        # --- 策略头 (保持不变) ---
        self.conv_policy1 = nn.Conv2d(channels, 128, kernel_size=3,padding=1, bias=False)
        self.bn_policy1 = nn.BatchNorm2d(128)
        self.conv_policy2 = nn.Conv2d(128, 32, kernel_size=1, bias=False)
        self.bn_policy2 = nn.BatchNorm2d(32)
        self.conv_policy3 = nn.Conv2d(32, policy_conv_out, kernel_size=1, bias=False)
        self.bn_policy3 = nn.BatchNorm2d(policy_conv_out)
        self.fc_policy1 = nn.Linear(policy_conv_out * board_size * board_size, policy_hidden)
        self.dropout_policy = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc_policy2 = nn.Linear(policy_hidden, board_size * board_size)

        # --- 价值头 (增强版) ---
        self.conv_value = nn.Conv2d(channels, value_channels, kernel_size=1, bias=False)
        self.bn_value = nn.BatchNorm2d(value_channels)
        self.gap_value = nn.AdaptiveAvgPool2d((3, 3))

        self.fc_value1 = nn.Linear(value_channels * 3 * 3, value_hidden1)
        self.dropout_value1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc_value2 = nn.Linear(value_hidden1, value_hidden2)
        self.dropout_value2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc_value3 = nn.Linear(value_hidden2, 1)

    def forward(self, x):
        # 共享特征提取
        out = F.relu(self.bn_input(self.conv_input(x)))
        for res_block in self.res_blocks:
            out = res_block(out)

        # 策略头 (逻辑不变)
        policy = F.relu(self.bn_policy1(self.conv_policy1(out)))
        policy = F.relu(self.bn_policy2(self.conv_policy2(policy)))
        policy = F.relu(self.bn_policy3(self.conv_policy3(policy)))
        policy = policy.view(policy.size(0), -1)
        policy = F.relu(self.fc_policy1(policy))
        policy = self.dropout_policy(policy)
        policy = self.fc_policy2(policy)

        # 价值头 (自动适配新的通道数)
        value = F.relu(self.bn_value(self.conv_value(out)))
        value = self.gap_value(value)
        value = value.view(value.size(0), -1)
        value = F.relu(self.fc_value1(value))
        value = self.dropout_value1(value)
        value = F.relu(self.fc_value2(value))
        value = self.dropout_value2(value)
        value = torch.tanh(self.fc_value3(value))

        return policy, value

    # save_model 和 load_model 方法保持不变，load_model 中的 strict=False 会自动处理结构变化
    def get_policy_value(self, state, legal_moves_mask=None, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            # 转换为2通道，并确保有 batch 维度
            if state.ndim == 3: # 单个样本 (C, H, W)
                state = state[:2] # 取前两个通道 (2, H, W)
                state = state[np.newaxis, :] # 添加 batch 维度 (1, 2, H, W)
            else: # 批次 (B, C, H, W)
                state = state[:, :2] # (B, 2, H, W)

            state_tensor = torch.from_numpy(state).float().to(device)

            if legal_moves_mask is not None:
                if legal_moves_mask.ndim == 1:
                    legal_moves_mask = legal_moves_mask[np.newaxis, :]
                legal_tensor = torch.from_numpy(legal_moves_mask).float().to(device)
            else:
                legal_tensor = None

            # 仅在 CUDA 时使用 autocast
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    policy_logits, value = self(state_tensor)
            else:
                policy_logits, value = self(state_tensor)

            if legal_tensor is not None:
                policy_logits = policy_logits.masked_fill(legal_tensor == 0, -1e4)

            policy = F.softmax(policy_logits, dim=1)
            value = value.cpu().numpy().flatten()

            policy = policy.cpu().numpy()
            return policy[0], value[0]

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'board_size': self.board_size,
            'channels': self.channels,
            'num_res_blocks': self.num_res_blocks,
            'dropout_rate': self.dropout_rate,
            'input_channels': self.input_channels,
        }, path)
        print(f"[模型] 已保存到 {path}")

    @staticmethod
    def load_model(path, device='cpu'):
        """加载模型，自动处理结构变化"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # 创建模型
        model = PolicyValueNet(
            board_size=checkpoint['board_size'],
            num_res_blocks=checkpoint.get('num_res_blocks', 6),
            channels=checkpoint.get('channels', 96),
            dropout_rate=checkpoint.get('dropout_rate', 0.1),
            input_channels=checkpoint.get('input_channels', 2)
        )

        old_state_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()

        # 统计信息
        loaded_count = 0
        skipped_count = 0
        mismatched_layers = []

        # 构建新的state_dict，只包含匹配的层
        new_state_dict = {}

        for name, param in old_state_dict.items():
            if name in model_dict:
                if param.shape == model_dict[name].shape:
                    new_state_dict[name] = param
                    loaded_count += 1
                else:
                    mismatched_layers.append(f"{name}: {param.shape} vs {model_dict[name].shape}")
                    skipped_count += 1
            else:
                mismatched_layers.append(f"{name}: not in current model")
                skipped_count += 1

        # 打印加载信息
        print(f"[模型加载] 成功加载 {loaded_count} 层, 跳过 {skipped_count} 层")

        if mismatched_layers:
            print(f"[警告] 跳过的层 (前10个):")
            for layer in mismatched_layers[:10]:
                print(f"  - {layer}")
            if len(mismatched_layers) > 10:
                print(f"  ... 还有 {len(mismatched_layers) - 10} 个")

        # 加载匹配的层
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        print(f"[模型] 已从 {path} 加载")
        return model
def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 测试模型
    model = PolicyValueNet(dropout_rate=0.1)
    total_params = count_parameters(model)
    print(f"模型总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 测试前向传播（2通道输入）
    x = torch.randn(1, 2, 15, 15)
    policy, value = model(x)
    print(f"策略输出形状: {policy.shape}")
    print(f"价值输出形状: {value.shape}")

    # 测试 get_policy_value（输入4通道）
    state = np.random.randn(4, 15, 15).astype(np.float32)
    policy_prob, value_est = model.get_policy_value(state)
    print(f"策略概率维度: {len(policy_prob)}")
    print(f"价值估计: {value_est:.4f}")
