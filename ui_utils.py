"""
GUI工具模块 - 提供通用的UI组件和绘制函数
"""
import tkinter as tk
from tkinter import ttk
import numpy as np


class GameBoard:
    """棋盘绘制类"""

    def __init__(self, canvas, board_size, cell_size, margin):
        """
        初始化棋盘
        Args:
            canvas: Tkinter Canvas对象
            board_size: 棋盘大小
            cell_size: 格子大小（像素）
            margin: 边距（像素）
        """
        self.canvas = canvas
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = margin
        self.board_width = (board_size - 1) * cell_size

    def draw_grid(self):
        """绘制网格线"""
        for i in range(self.board_size):
            y = self.margin + i * self.cell_size
            self.canvas.create_line(self.margin, y, self.margin + self.board_width, y,
                                    fill='black', width=1)
            x = self.margin + i * self.cell_size
            self.canvas.create_line(x, self.margin, x, self.margin + self.board_width,
                                    fill='black', width=1)

    def draw_stars(self):
        """绘制星位"""
        star_positions = []
        if self.board_size == 15:
            star_positions = [(3, 3), (11, 3), (7, 7), (3, 11), (11, 11)]
        elif self.board_size == 19:
            star_positions = [(3, 3), (15, 3), (9, 9), (3, 15), (15, 15)]

        for r, c in star_positions:
            x = self.margin + c * self.cell_size
            y = self.margin + r * self.cell_size
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill='black', outline='black')

    def draw_piece(self, row, col, piece):
        """
        绘制棋子
        Args:
            row: 行坐标
            col: 列坐标
            piece: 棋子类型 (1: 黑棋, -1: 白棋, 0: 空)
        """
        if piece == 0:
            return

        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        radius = self.cell_size // 2 - 2
        color = 'black' if piece == 1 else 'white'
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                fill=color, outline='gray')

    def highlight_move(self, row, col):
        """高亮最后一步"""
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill='red', outline='red')

    def draw_stat_circle(self, row, col, value, max_value, text):
        """
        绘制统计圆圈（用于MCTS统计）- 连续渐变颜色，浅色背景
        Args:
            row: 行坐标
            col: 列坐标
            value: 当前值（用于决定颜色）
            max_value: 最大值（用于归一化）
            text: 显示的文本
        """
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        radius = self.cell_size // 2.5

        # 归一化值 (0-1)
        norm_value = value
        norm_value = max(0.0, min(1.0, norm_value))

        # 连续渐变颜色 - 使用更浅的颜色
        # 从浅红色(0) -> 浅黄色(0.5) -> 浅绿色(1)
        # 基础亮度提高，颜色更柔和
        if norm_value <= 0.5:
            # 浅红色到浅黄色：R=255, G从180到255线性增加
            r = 255
            g = int(100 + 75 * (norm_value / 0.5))
            b = 100
        else:
            # 浅黄色到浅绿色：G=255, R从255减少到180
            r = int(255 - 75 * ((norm_value - 0.5) / 0.5))
            g = 255
            b = 100

        # 转换为十六进制颜色
        color = f'#{r:02x}{g:02x}{b:02x}'

        # 绘制背景圆（不透明，浅色）
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                fill=color, outline='')

        # 绘制文本 - 居中显示
        self.canvas.create_text(x, y, text=text, font=('Arial', 9, 'bold'),
                                fill='black', justify='center')

    def draw_best_circle(self, row, col):
        """绘制最佳点绿色圆圈"""
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        radius = self.cell_size // 2 - 2 + 4
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                outline='green', width=2, fill='')

    def coord_to_index(self, x, y):
        """
        将画布坐标转换为棋盘坐标
        Args:
            x: 画布x坐标
            y: 画布y坐标
        Returns:
            (row, col) 或 None（如果点击在棋盘外）
        """
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)

        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return (row, col)
        return None





class StatsBoard:
    """统计棋盘类 - 用于显示MCTS统计"""

    def __init__(self, canvas, board_size, cell_size, margin):
        """
        初始化统计棋盘
        Args:
            canvas: Tkinter Canvas对象
            board_size: 棋盘大小
            cell_size: 格子大小（像素）
            margin: 边距（像素）
        """
        self.board = GameBoard(canvas, board_size, cell_size, margin)
        self.board_size = board_size

    def draw_empty_board(self):
        """绘制空棋盘"""
        self.board.canvas.delete("all")
        self.board.draw_grid()
        self.board.draw_stars()

    def draw_stats(self, game_board, mcts_root, display_mode):
        """
        绘制统计信息
        Args:
            game_board: 游戏棋盘状态（用于判断哪些位置有棋子）
            mcts_root: MCTS根节点
            display_mode: 显示模式 ("MCTS统计" 或 "策略网络概率")
        """
        self.draw_empty_board()

        if not mcts_root or not mcts_root.children:
            return

        # 先绘制已有棋子（半透明显示）
        for r in range(self.board_size):
            for c in range(self.board_size):
                piece = game_board[r, c]
                if piece != 0:
                    # 绘制半透明棋子
                    x = self.board.margin + c * self.board.cell_size
                    y = self.board.margin + r * self.board.cell_size
                    radius = self.board.cell_size // 2 - 2
                    color = 'gray' if piece == 1 else 'lightgray'
                    self.board.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                                  fill=color, outline='gray', stipple='gray50')

        # 收集统计数据
        stats = []
        max_visits = 0
        max_winrate = 0
        max_prob = 0

        for move, child in mcts_root.children.items():
            r, c = move
            if game_board[r, c] != 0:
                continue

            visits = child.visit_count
            winrate = (1 - child.get_value()) / 2 if visits > 0 else 0.5
            prob = child.prior_prob

            stats.append((move, visits, winrate, prob))
            max_visits = max(max_visits, visits)
            max_winrate = max(max_winrate, winrate)
            max_prob = max(max_prob, prob)

        # 找出最佳点
        best_move = None
        if display_mode == "MCTS统计" and max_visits > 0:
            best_move = max(stats, key=lambda x: x[1])[0]
        elif display_mode == "策略网络概率" and max_prob > 0:
            best_move = max(stats, key=lambda x: x[3])[0]

        # 绘制统计信息
        for move, visits, winrate, prob in stats:
            r, c = move

            if display_mode == "MCTS统计":
                if visits == 0:
                    continue
                value = winrate
                max_value = max_winrate
                text = f"{visits}\n{winrate * 100:.0f}%"
            else:  # 策略网络概率
                value = prob
                max_value = max_prob
                text = f"{prob * 100:.1f}%"

            self.board.draw_stat_circle(r, c, value, max_value, text)

        # 绘制最佳点绿圈
        if best_move is not None:
            r, c = best_move
            self.board.draw_best_circle(r, c)
class TrainingStatsPanel:
    """训练状态面板 - 整合统计和损失显示"""

    def __init__(self, parent):
        self.parent = parent
        self.stats_label = None
        self.current_game_label = None
        self.policy_loss_label = None
        self.value_loss_label = None
        self.entropy_loss_label = None
        self.total_loss_label = None
        self.lr_display_label = None

    def create(self):
        stats_frame = tk.Frame(self.parent)
        stats_frame.pack(fill=tk.X, pady=5)
        self.stats_label = tk.Label(stats_frame, text="等待开始...", font=('微软雅黑', 10))
        self.stats_label.pack(side=tk.LEFT)
        self.current_game_label = tk.Label(stats_frame, text="", font=('微软雅黑', 10), foreground='blue')
        self.current_game_label.pack(side=tk.RIGHT)

        loss_frame = tk.Frame(self.parent)
        loss_frame.pack(fill=tk.X, pady=5)
        self.policy_loss_label = tk.Label(loss_frame, text="策略损失: --", font=('微软雅黑', 9))
        self.policy_loss_label.pack(side=tk.LEFT, padx=5)
        self.value_loss_label = tk.Label(loss_frame, text="价值损失: --", font=('微软雅黑', 9))
        self.value_loss_label.pack(side=tk.LEFT, padx=5)
        self.entropy_loss_label=tk.Label(loss_frame,text="熵损失: --",font=('微软雅黑',9))
        self.entropy_loss_label.pack(side=tk.LEFT, padx=5)
        self.total_loss_label = tk.Label(loss_frame, text="总损失: --", font=('微软雅黑', 9))
        self.total_loss_label.pack(side=tk.LEFT, padx=5)
        self.lr_display_label = tk.Label(loss_frame, text="学习率: --", font=('微软雅黑', 9))
        self.lr_display_label.pack(side=tk.RIGHT, padx=5)

    def update_stats(self, data_size, game_count, train_count, lr):
        self.stats_label.config(text=f"数据量: {data_size} 条 | 对局数: {game_count} | 训练步数: {train_count}")
        self.lr_display_label.config(text=f"学习率: {lr:.6f}")

    def update_loss(self, policy_loss, value_loss,entropy_loss, total_loss):
        self.policy_loss_label.config(text=f"策略损失: {policy_loss:.4f}")
        self.value_loss_label.config(text=f"价值损失: {value_loss:.4f}")
        self.entropy_loss_label.config(text=f"熵损失: {entropy_loss:.4f}")
        self.total_loss_label.config(text=f"总损失: {total_loss:.4f}")

    def update_current_game(self, game_id):
        self.current_game_label.config(text=f"当前: #{game_id}")

