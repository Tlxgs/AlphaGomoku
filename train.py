"""
自我对弈训练模块 - 单线程模式，支持实时对局显示
"""
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import random
import os
import pickle
import glob
from datetime import datetime
import models
from game import GomokuGame
from mcts import MCTS
from play.gomoku.model import PolicyValueNet

os.environ["TORCHDYNAMO_BACKEND"] = "eager"
torch._dynamo.config.suppress_errors = True
class TrainingData:
    """训练数据缓冲区，支持按时间戳保存多个文件"""
    def __init__(self, max_size=50000, data_dir='data/'):
        self.max_size = max_size
        self.data_dir = data_dir
        self.data = deque(maxlen=max_size)
        os.makedirs(data_dir, exist_ok=True)

    def add(self, state, policy, value):
        self.data.append((state, policy, value))

    def add_game(self, states, policies, players, winner):
        """将一局游戏添加到数据缓冲区"""
        num_moves = len(states)

        for i, (state, policy, player) in enumerate(zip(states, policies, players)):
            if winner == 0:
                value = 0.0
            else:
                if winner == player:
                    value = 1.0
                else:
                    value = -1.0

            self.add(state, policy, value)

    def sample(self, batch_size):
        batch = random.sample(list(self.data), min(batch_size, len(self.data)))
        states, policies, values = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(policies, dtype=np.float32),
                np.array(values, dtype=np.float32))

    def __len__(self):
        return len(self.data)

    def save_game(self, game_data):
        """
        保存单局游戏数据到新文件
        game_data: 包含 (states, policies, players, winner) 的元组
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"data_{timestamp}.pkl"
        filepath = os.path.join(self.data_dir, filename)

        # 只保存这一局的数据
        states, policies, players, winner = game_data
        save_info = {
            'states': states,
            'policies': policies,
            'players': players,
            'winner': winner,
            'timestamp': datetime.now().isoformat(),
            'num_moves': len(states)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_info, f)
        print(f"[数据] 已保存第 {len(glob.glob(os.path.join(self.data_dir, 'data_*.pkl')))} 局到 {filepath}")

    def load_all(self):
        pattern = os.path.join(self.data_dir, "data_*.pkl")
        files = glob.glob(pattern)
        if not files:
            return False

        files.sort(key=os.path.getmtime, reverse=True)

        all_data = []  # 临时存放所有数据
        loaded_files = []  # 记录成功加载的文件

        for f in files:
            try:
                with open(f, 'rb') as fp:
                    save_info = pickle.load(fp)
                states = save_info['states']
                policies = save_info['policies']
                players = save_info['players']
                winner = save_info['winner']
                num_moves = len(states)

                for i, (state, policy, player) in enumerate(zip(states, policies, players)):
                    steps_to_end = num_moves - i
                    if winner == 0:
                        value = 0.0
                    else:
                        value = 1.0 if winner == player else -1.0
                    all_data.append((state, policy, value))

                    if len(all_data) >= self.max_size:
                        break

                loaded_files.append(f)
                if len(all_data) >= self.max_size:
                    break
            except Exception as e:
                raise RuntimeError(f"读取文件 {f} 失败: {e}")

        # 清空原有数据，填入新数据
        self.data.clear()
        for item in all_data[:self.max_size]:
            self.data.append(item)

        print(f"[数据] 自动加载完成: {len(self.data)} 条数据 (来自 {len(loaded_files)} 个文件)")
        return True

    def load_single(self, filename):
        """加载单个数据文件"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            return False

        with open(filepath, 'rb') as f:
            save_info = pickle.load(f)

        # 清空当前数据
        self.data.clear()

        # 加载这一局的数据
        states = save_info['states']
        policies = save_info['policies']
        players = save_info['players']
        winner = save_info['winner']

        num_moves = len(states)
        for i, (state, policy, player) in enumerate(zip(states, policies, players)):
            steps_to_end = num_moves - i

            if winner == 0:
                value = 0.0
            else:
                if winner == player:
                    value = 1.0
                else:
                    value = -1.0

            self.add(state, policy, value)

        print(f"[数据] 已加载 {len(self.data)} 条数据从 {filename}")
        return True


class GameResult:
    """游戏结果记录（用于UI显示）"""
    def __init__(self, game_id, board_size, moves, winner):
        self.game_id = game_id
        self.board_size = board_size
        self.moves = moves  # [(row, col, player), ...]
        self.winner = winner


def play_one_game(model=None, board_size=15, num_simulations=100, device='cuda', game_id=None,
                  c_puct=3, temperature=1.0, exploration_mode=False, onnx_path='model.onnx'):
    # 创建 MCTS
    mcts = MCTS(
        model=None,
        c_puct=c_puct,
        num_simulations=num_simulations,
        temperature=temperature,
        onnx_path=onnx_path,
        device='cuda'
    )

    game = GomokuGame(board_size)
    states, policies, players = [], [], []
    moves = []  # 记录落子顺序
    move_count = 0
    mcts.root = None
    # 探索模式：开局随机放置一黑两白
    if exploration_mode:
        empty_positions = game.get_legal_moves()
        if len(empty_positions) >= 3:
            # 随机选择三个不同位置
            idx1, idx2,idx3 = random.sample(range(len(empty_positions)), 3)
            pos1 = empty_positions[idx1]
            pos2 = empty_positions[idx2]
            pos3 = empty_positions[idx3]
            # 先放黑子
            game.make_move(pos1[0], pos1[1])
            moves.append((pos1[0], pos1[1], 1))
            move_count += 1

            # 再放白子
            game.make_move(pos2[0], pos2[1])
            moves.append((pos2[0], pos2[1], -1))
            move_count += 1
            game.switch_player()
            game.make_move(pos3[0], pos3[1])
            moves.append((pos3[0], pos3[1], -1))
            move_count += 1


            # 初始化MCTS根节点到当前游戏状态
            if mcts:
                mcts.init_root(game)
    # 强制第一步下天元
    天地大同 = False
    if 天地大同:
        center = board_size // 2
        if game.is_legal_move(center, center):
            state = game.get_canonical_state()
            policy = np.zeros(board_size * board_size, dtype=np.float32)
            idx = center * board_size + center
            policy[idx] = 1.0
            states.append(state)
            policies.append(policy)
            players.append(game.current_player)
            moves.append((center, center, game.current_player))
            game.make_move(center, center)
            if mcts:
                mcts.init_root(game)
            move_count += 1
    while not game.game_over:
        # 动态温度
        temp = max([0.12, temperature * 0.94 ** move_count])
        if game.current_player != 1:
            temp=temp*0.8
        move_probs, _ = mcts.get_move_probs(game, temp)

        state = game.get_canonical_state()
        policy = np.zeros(board_size * board_size, dtype=np.float32)
        for move, prob in move_probs.items():
            idx = move[0] * board_size + move[1]
            policy[idx] = prob

        states.append(state)
        policies.append(policy)
        players.append(game.current_player)

        if move_probs:
            moves_list = list(move_probs.keys())
            probs = list(move_probs.values())
            selected_move = moves_list[np.random.choice(len(moves_list), p=probs)]

            moves.append((selected_move[0], selected_move[1], game.current_player))

            game.make_move(selected_move[0], selected_move[1])
            mcts.update_root(game, selected_move)

        move_count += 1
        print(move_count)

    return states, policies, players, moves, game.winner


class SelfPlayTrainer:
    """自我对弈训练器（单线程，支持实时对局显示）"""

    def __init__(self, model_path='model.pt', board_size=15, device='cuda',
                 num_simulations=100, data_dir='data/',
                 use_rollout=False, rollout_alpha=0.6, max_rollout_depth=8,
                 c_puct=5, learning_rate=0.00025, weight_decay=1e-4):
        self.model_path = model_path
        self.board_size = board_size
        self.device = device
        self.num_simulations = num_simulations
        self.data_dir = data_dir
        self.use_rollout = use_rollout
        self.rollout_alpha = rollout_alpha
        self.max_rollout_depth = max_rollout_depth
        self.c_puct = c_puct
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        os.makedirs(data_dir, exist_ok=True)
        self.model = PolicyValueNet.load_model(self.model_path).to(self.device)
        self.data_buffer = TrainingData(max_size=40000, data_dir=data_dir)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)

        # 学习率调度
        warmup_epochs = 10
        def warmup_lambda(epoch):
            return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                              lr_lambda=warmup_lambda)

        self.game_count = 0
        self.train_count = 0

        # 预计算数据增强索引
        indices = np.arange(board_size * board_size).reshape(board_size, board_size)
        self.flip_ud_map = np.flipud(indices).flatten()
        self.flip_lr_map = np.fliplr(indices).flatten()
        self.flip_both_map = np.flipud(np.fliplr(indices)).flatten()
        self.rot90_map = np.rot90(indices, k=1).flatten()
        self.rot270_map = np.rot90(indices, k=3).flatten()
        self.transpose_map = indices.T.flatten()  # 主对角线
        self.anti_transpose_map = np.rot90(indices, k=1).T.flatten()  # 副对角线

    def _augment_batch(self, states, policies):
        batch_size = states.shape[0]
        aug_states = []
        aug_policies = []
        sym_transforms = [
            (lambda x: x, None, None),  # 0: 恒等
            (lambda x: np.flip(x, axis=1), self.flip_ud_map, None),  # 1: 上下翻转
            (lambda x: np.flip(x, axis=2), self.flip_lr_map, None),  # 2: 左右翻转
            (lambda x: np.flip(x, axis=(1, 2)), self.flip_both_map, None),  # 3: 旋转180°
            (lambda x: np.rot90(x, k=1, axes=(1, 2)), self.rot90_map, None),  # 4: 顺时针90°
            (lambda x: np.rot90(x, k=3, axes=(1, 2)), self.rot270_map, None),  # 5: 逆时针90°
            (lambda x: np.transpose(x, (0, 2, 1)), self.transpose_map, None),  # 6: 主对角线翻转
            (lambda x: np.flip(np.transpose(x, (0, 2, 1)), axis=2), self.anti_transpose_map, None)  # 7: 副对角线翻转
        ]

        for i in range(batch_size):
            state = states[i]
            policy = policies[i]
            sym_idx = np.random.randint(0, 8)
            transform_func, policy_map, _ = sym_transforms[sym_idx]
            # 应用几何变换
            aug_state = transform_func(state)
            if policy_map is not None:
                aug_policy = policy[policy_map]
            else:
                aug_policy = policy

            aug_states.append(aug_state)
            aug_policies.append(aug_policy)

        return (np.array(aug_states, dtype=np.float32),
                np.array(aug_policies, dtype=np.float32))
    def generate_one_game(self, temperature=1.0):
        """生成一局游戏"""
        states, policies, players, moves, winner = play_one_game(
            self.model, self.board_size, self.num_simulations, self.device,
            self.game_count + 1, self.use_rollout, self.rollout_alpha,
            self.max_rollout_depth, self.c_puct, temperature
        )
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return states, policies, players, moves, winner

    def train_step(self, batch_size=1024):
        """执行一步训练"""
        if len(self.data_buffer) < batch_size:
            return None, None, None, None

        states, policies, values = self.data_buffer.sample(batch_size)
        states, policies= self._augment_batch(states, policies)
        states = states[:,:2]
        states_tensor = torch.FloatTensor(states).to(self.device)
        policies_tensor = torch.FloatTensor(policies).to(self.device)
        values_tensor = torch.FloatTensor(values).unsqueeze(1).to(self.device)

        self.model.train()
        policy_logits, pred_values = self.model(states_tensor)

        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = - (policies_tensor * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(pred_values, values_tensor )

        # ========== 新增：熵损失 ==========
        probs = F.softmax(policy_logits, dim=1)
        entropy = - (probs * log_probs).sum(dim=1).mean()  # 熵值
        entropy_weight = 0.1  # 熵损失权重（可调）
        entropy_loss = -entropy_weight * entropy  # 负号：最大化熵
        # ==================================
        total_loss = policy_loss + value_loss + entropy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(),entropy_loss.item(), total_loss.item()

    def save_model(self):
        self.model.save_model(self.model_path)

    def load_training_data(self, filename=None):
        """加载训练数据，filename 为 None 时自动加载所有最新文件"""
        if filename is not None:
            return self.data_buffer.load_single(filename)
        else:
            return self.data_buffer.load_all()

    def save_training_data(self, game_data):
        """保存单局训练数据到新文件"""
        self.data_buffer.save_game(game_data)

    def update_hyperparameters(self, **kwargs):
        """更新超参数"""
        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        if 'weight_decay' in kwargs:
            self.weight_decay = kwargs['weight_decay']
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = self.weight_decay
        if 'c_puct' in kwargs:
            self.c_puct = kwargs['c_puct']
        if 'num_simulations' in kwargs:
            self.num_simulations = kwargs['num_simulations']
        if 'use_rollout' in kwargs:
            self.use_rollout = kwargs['use_rollout']
        if 'rollout_alpha' in kwargs:
            self.rollout_alpha = kwargs['rollout_alpha']
        if 'max_rollout_depth' in kwargs:
            self.max_rollout_depth = kwargs['max_rollout_depth']
