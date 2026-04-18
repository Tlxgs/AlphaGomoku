"""
五子棋游戏逻辑
"""
import numpy as np

class GomokuGame:
    """五子棋游戏类"""

    def __init__(self, board_size=15):
        self.board_size = board_size
        self.reset()

    def reset(self):
        """重置游戏"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 1: 黑棋, -1: 白棋
        self.last_move = None
        self.game_over = False
        self.winner = None
        self.move_count = 0
        return self.get_state()

    def get_state(self):
        """获取当前状态（用于神经网络输入）"""
        # 4通道: 当前玩家棋子, 对手棋子, 最后一步, 当前玩家标识
        state = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)
        
        # 当前玩家的棋子
        state[0] = (self.board == self.current_player).astype(np.float32)
        # 对手的棋子
        state[1] = (self.board == -self.current_player).astype(np.float32)
        # 最后一步
        if self.last_move is not None:
            state[2, self.last_move[0], self.last_move[1]] = 1.0
        # 当前玩家标识
        state[3] = (self.current_player + 1) / 2.0  # 0或1
        
        return state
    
    def get_canonical_state(self, player=None):
        """获取规范状态（从当前玩家视角）"""
        if player is None:
            player = self.current_player
        
        state = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)
        state[0] = (self.board == player).astype(np.float32)
        state[1] = (self.board == -player).astype(np.float32)
        if self.last_move is not None:
            state[2, self.last_move[0], self.last_move[1]] = 1.0
        state[3] = (player + 1) / 2.0
        
        return state
    
    def get_legal_moves(self):
        """获取合法落子位置"""
        return list(zip(*np.where(self.board == 0)))
    
    def get_legal_moves_mask(self):
        """获取合法落子掩码"""
        return (self.board == 0).flatten().astype(np.float32)
    
    def is_legal_move(self, row, col):
        """检查落子是否合法"""
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        return self.board[row, col] == 0 and not self.game_over

    def unsafe_make_move(self, row, col):
        """跳过合法性检查的落子（仅用于 MCTS 模拟）"""
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.move_count += 1

        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif self.move_count >= self.board_size * self.board_size:
            self.game_over = True
            self.winner = 0
        else:
            self.current_player = -self.current_player
        return True
    def make_move(self, row, col):
        """落子"""
        if not self.is_legal_move(row, col):
            return False
        
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.move_count += 1
        
        # 检查胜负
        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif self.move_count >= self.board_size * self.board_size:
            self.game_over = True
            self.winner = 0  # 平局
        else:
            self.current_player = -self.current_player
        
        return True
    def switch_player(self):
        self.current_player = - self.current_player
    def check_win(self, row, col):
        """检查是否获胜"""
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # 正方向
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            # 反方向
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                return True
        
        return False
    
    def get_result(self, player):
        """获取游戏结果（从指定玩家视角）"""
        if not self.game_over:
            return None
        if self.winner == 0:
            return 0.0  # 平局
        return 1.0 if self.winner == player else -1.0
    
    def copy(self):
        """复制游戏状态"""
        new_game = GomokuGame(self.board_size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.last_move = self.last_move
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.move_count = self.move_count
        return new_game
    
    def get_nearby_moves(self, distance=2):
        """获取已有棋子附近的合法位置（用于剪枝）"""
        if self.move_count == 0:
            # 第一步下中心
            center = self.board_size // 2
            return [(center, center),(center+1,center),(center-1,center),
                    (center,center-1),(center,center+1),(center+1,center+1),
                    (center-1,center-1),(center-1,center+1),(center+1,center-1),]
        
        nearby = set()
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] != 0:
                    for dr in range(-distance, distance + 1):
                        for dc in range(-distance, distance + 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                                if self.board[nr, nc] == 0:
                                    nearby.add((nr, nc))
        
        return list(nearby) if nearby else self.get_legal_moves()
    
    def __str__(self):
        """打印棋盘"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        s = "   " + " ".join(f"{i:2d}" for i in range(self.board_size)) + "\n"
        for i in range(self.board_size):
            s += f"{i:2d} " + "  ".join(symbols[self.board[i, j]] for j in range(self.board_size)) + "\n"
        return s
