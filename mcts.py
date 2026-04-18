""" 蒙特卡洛树搜索 (MCTS)"""
import numpy as np
import math
import os
import time
from typing import Dict, Tuple
import onnxruntime as ort
def _is_winning_move(game, row, col):
    """检查在 (row, col) 落子后当前玩家是否获胜（不修改游戏状态）"""
    if game.board[row, col] != 0:
        return False
    player = game.current_player
    original = game.board[row, col]
    game.board[row, col] = player
    win = game.check_win(row, col)
    game.board[row, col] = original
    return win
def _get_blocking_moves(game):
    """
    获取所有能阻挡对手直接获胜的落子位置
    即：对手在这些位置落子就会赢，所以当前玩家必须占住这些位置
    """
    current_player = game.current_player
    opponent = -current_player
    blocking_moves = []

    legal_moves = game.get_legal_moves()

    for move in legal_moves:
        row, col = move

        # 检查如果对手在这里落子，是否会赢
        original = game.board[row, col]
        game.board[row, col] = opponent
        win = game.check_win(row, col)
        game.board[row, col] = original

        if win:
            blocking_moves.append(move)

    return blocking_moves
class MCTSNode:
    """MCTS节点"""
    def __init__(self, parent=None, prior_prob=1.0):
        self.parent = parent
        self.children: Dict[Tuple[int, int], 'MCTSNode'] = {}
        self.prior_prob = prior_prob
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False

    def is_leaf(self):
        return not self.expanded or not self.children  # 展开但无子节点也视为叶子

    def get_value(self,default=0.0):
        if self.visit_count == 0:
            return default
        return self.value_sum / self.visit_count
    def get_opposite_value(self,default=0.0):
        return -self.get_value(-default)

    def select_child(self, game, c_puct):
        if not self.children:
            return None, None

        explore_penalty_value = 0.05
        parent_value = self.get_value() - explore_penalty_value
        sqrt_parent_visits = math.sqrt(1 + self.visit_count)

        best_score = -float('inf')
        best_move = None
        best_child = None
        for move, child in self.children.items():
            q_val = -child.get_value(-parent_value)
            u_val = c_puct * child.prior_prob * sqrt_parent_visits / (1 + child.visit_count)
            score = q_val + u_val

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child
    def expand(self, move_probs: Dict[Tuple[int, int], float]):
        self.expanded = True
        for move, prob in move_probs.items():
            self.children[move] = MCTSNode(parent=self, prior_prob=prob)


    def update(self, value):
        self.visit_count += 1
        self.value_sum += value

    def update_recursive(self, value):
        """从当前节点开始递归向上更新（自动翻转符号）"""
        self.update(value)
        if self.parent is not None:
            self.parent.update_recursive(-value)

    def prune_illegal_children(self, game):
        """删除所有在当前游戏状态下不合法的子节点"""
        illegal_moves = []
        for move in list(self.children.keys()):
            if not game.is_legal_move(move[0], move[1]):
                illegal_moves.append(move)
        for move in illegal_moves:
            del self.children[move]
        # 如果所有子节点都被删除，标记为未展开（下次会重新展开）
        if not self.children:
            self.expanded = False


class MCTS:
    def __init__(self, model=None, c_puct=5, num_simulations=100, temperature=1.0,
                 onnx_path='model.onnx',device='cpu'):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.root = None
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
        self.board_size = 15
        self.input_buffer = np.zeros((1, 2, 15, 15), dtype=np.float32)
        self.policy_buffer = np.zeros((1, 225), dtype=np.float32)
        self.value_buffer = np.zeros((1, 1), dtype=np.float32)
        if device=='cpu':
            providers = ['CPUExecutionProvider']
            print("使用CPU推理")
        if device=='cuda':
            providers = ['CUDAExecutionProvider']
            print("使用GPU推理")
        self.ort_session = ort.InferenceSession(onnx_path, sess_options, providers=providers)


    def _onnx_inference(self, state, legal_mask):
        """ONNX 推理优化版 - 减少内存分配"""
        self.input_buffer[0, 0] = state[0]
        self.input_buffer[0, 1] = state[1]
        outputs = self.ort_session.run(
            None,
            {'input': self.input_buffer}
        )
        policy_logits = outputs[0][0]
        value = outputs[1][0, 0]
        if legal_mask is not None:
            policy_logits[legal_mask == 0] = -1e8
        max_logit = policy_logits.max()
        np.exp(policy_logits - max_logit, out=policy_logits)
        policy_logits /= policy_logits.sum()

        return policy_logits, value
    def update_root(self, game, last_move):
        if self.root is None:
            return
        if last_move in self.root.children:
            new_root = self.root.children[last_move]
            new_root.parent = None
            self.root = new_root
        else:
            self.root = None

    def init_root(self, game):
        """显式初始化根节点（通常用于游戏开始或重置）"""
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            self.root = None
            return

        # 获取网络评估
        state = game.get_canonical_state()
        legal_mask = game.get_legal_moves_mask()
        policy, _ = self._onnx_inference(state, legal_mask)

        move_probs_all = {}
        for i, move in enumerate(legal_moves):
            idx = move[0] * game.board_size + move[1]
            prob = policy[idx]
            move_probs_all[move] = prob

        # 归一化
        total_prob = sum(move_probs_all.values())
        if total_prob > 0:
            for move in move_probs_all:
                move_probs_all[move] /= total_prob
        else:
            prob = 1.0 / len(legal_moves)
            move_probs_all = {move: prob for move in legal_moves}

        # 创建根节点
        self.root = MCTSNode()
        self.root.expanded = True
        for move, prob in move_probs_all.items():
            self.root.children[move] = MCTSNode(parent=self.root, prior_prob=prob)

    def get_move_probs(self, game, temp=None):
        """获取落子概率（核心决策函数）"""
        if temp is None:
            temp = self.temperature
        legal_moves = game.get_legal_moves()
        winning_moves = []
        for move in legal_moves:
            if _is_winning_move(game, move[0], move[1]):
                winning_moves.append(move)

        if winning_moves:
            # 有直接获胜，返回获胜走法
            if len(winning_moves) == 1:
                best_move = winning_moves[0]
                return {best_move: 1.0}, best_move
            else:
                # 多个获胜走法，均匀分配概率
                prob = 1.0 / len(winning_moves)
                probs = {move: prob for move in winning_moves}
                best_move = winning_moves[0]  # 任意选一个
                return probs, best_move
        blocking_moves = _get_blocking_moves(game)
        if blocking_moves:
            if len(blocking_moves) == 1:
                best_move = blocking_moves[0]
                return {best_move: 1.0}, best_move
            else:
                # 多个阻挡点，均匀分配概率
                prob = 1.0 / len(blocking_moves)
                probs = {move: prob for move in blocking_moves}
                best_move = blocking_moves[0]
                return probs, best_move
        if len(legal_moves) == 0:
            return {}, None
        if len(legal_moves) == 1:
            return {legal_moves[0]: 1.0}, legal_moves[0]

        if self.root is None:
            # 初始化根节点（包含网络推理 + Dirichlet噪声）
            # 获取网络评估
            state = game.get_canonical_state()
            legal_mask = game.get_legal_moves_mask()
            policy, _ = self._onnx_inference(state, legal_mask)


            move_probs_all = {}
            for i, move in enumerate(legal_moves):
                idx = move[0] * game.board_size + move[1]
                prob =  policy[idx]
                move_probs_all[move] = prob

            # 归一化
            total_prob = sum(move_probs_all.values())
            if total_prob > 0:
                for move in move_probs_all:
                    move_probs_all[move] /= total_prob

            # 创建根节点并填充子节点
            self.root = MCTSNode()
            self.root.expanded = True
            for move, prob in move_probs_all.items():
                self.root.children[move] = MCTSNode(parent=self.root, prior_prob=prob)

        else:
            self.root.prune_illegal_children(game)
            if not self.root.children:
                self.root = None
                return self.get_move_probs(game, temp)  # 递归重新初始化

        # ---------- 执行MCTS模拟 ----------
        for _ in range(self.num_simulations):
            game_copy = game.copy()
            self._simulate(game_copy)

        # ---------- 根据访问次数计算策略 ----------
        move_visits = {}
        for move, child in self.root.children.items():
            move_visits[move] = child.visit_count

        if temp <= 0.1:
            best_move = max(move_visits.keys(), key=lambda m: move_visits[m])
            return {best_move: 1.0}, best_move
        else:
            moves = list(move_visits.keys())
            visits = np.array([move_visits[m] for m in moves])
            probs = visits ** (1.0 / temp)
            probs = probs / probs.sum()
            final_probs = {m: p for m, p in zip(moves, probs)}
            best_move = moves[np.argmax(probs)]
            return final_probs, best_move

    def _simulate(self, game):
        """执行一次模拟（选择、扩展、评估、反向传播）"""
        node = self.root
        while not node.is_leaf():
            move, child = node.select_child(game, self.c_puct)
            if move is None:
                value = 0
                node.update_recursive(value)
                return
            game.unsafe_make_move(move[0], move[1])
            node = child
        if game.game_over:
            if game.winner == 0:
                value = 0.0
            else:
                value = -1
            node.update_recursive(value)
            return
        state = game.get_canonical_state()
        legal_mask = game.get_legal_moves_mask()
        policy, value_net = self._onnx_inference(state, legal_mask)
        move_probs_all = {move: policy[move[0] * game.board_size + move[1]]
                          for move in game.get_legal_moves()}
        node.expand(move_probs_all)
        node.update_recursive(value_net)


    def get_best_move(self, game):
        """获取最佳落子位置"""
        _, best_move = self.get_move_probs(game, temp=0)
        return best_move

    def get_move_details(self, game):
        """获取每个合法落子的详细信息（访问次数和价值）"""
        if self.root is None:
            self.root = MCTSNode()
        self.get_move_probs(game, temp=1)  # 触发模拟
        details = {}
        for move, child in self.root.children.items():
            details[move] = (child.visit_count, child.get_value())
        return details