# train_gui.py
"""
五子棋AI训练GUI - 带棋盘显示对局和超参数调整
支持多进程自我对弈，每局自动保存数据，手动回放对局（支持上一步/下一步）
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import queue
import multiprocessing as mp
import os
import pickle
from datetime import datetime
import numpy as np
from train import SelfPlayTrainer, play_one_game, GameResult
from ui_utils import GameBoard, TrainingStatsPanel
from model import PolicyValueNet

def worker_process(trainer_params, model_path, result_queue, stop_event, device='cuda'):
    # 根据 .pt 路径构造 .onnx 路径
    onnx_path = model_path.replace('.pt', '.onnx')
    if not os.path.exists(onnx_path):
        print(f"[Worker] ONNX 文件不存在: {onnx_path}，请先导出 ONNX 模型")
        return

    board_size = trainer_params['board_size']
    num_simulations = trainer_params['num_simulations']
    c_puct = trainer_params['c_puct']
    temperature = trainer_params['temperature']
    exploration_mode = trainer_params['exploration_mode']

    np.random.seed()

    while not stop_event.is_set():
        states, policies, players, moves, winner = play_one_game(
            model=None,
            board_size=board_size,
            num_simulations=num_simulations,
            device=device,
            game_id=None,
            c_puct=c_puct,
            temperature=temperature,
            exploration_mode=exploration_mode,
            onnx_path=onnx_path   # 关键：传递 ONNX 路径
        )
        result_queue.put((states, policies, players, moves, winner))

class TrainingGUI:
    """训练GUI主类 - 带棋盘显示和超参数调整，多进程对局"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("五子棋AI训练系统 - 实时对局显示（多进程）")
        self.root.geometry("1500x900")

        self.trainer = None
        self.training_processes = []
        self.is_training = False
        self.update_queue = queue.Queue()
        self.stop_event = mp.Event()

        self.game_history = []          # 存储所有对局结果
        self.current_history_index = -1 # 当前显示的对局索引
        self.current_game_moves = []    # 当前对局的落子列表
        self.current_winner = None
        self.current_game_id = None
        self.current_step = 0            # 当前显示的步数（0 表示空棋盘）
        self.model=PolicyValueNet.load_model('backup/model.pt').to('cuda')
        self._create_widgets()
        self.root.after(100, self._process_queue)
        self.root.mainloop()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.LabelFrame(main_frame, text="训练控制", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self._create_model_settings(left_frame)
        self._create_mcts_settings(left_frame)
        self._create_training_settings(left_frame)
        self._create_optimizer_settings(left_frame)
        self._create_data_settings(left_frame)
        self._create_buttons(left_frame)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._create_board_area(right_frame)
        self._create_status_area(right_frame)

        self._draw_empty_board()

    def _create_model_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="模型设置", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(frame, text="模型路径:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.model_path_var = tk.StringVar(value="model.pt")
        ttk.Entry(frame, textvariable=self.model_path_var, width=30).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(frame, text="浏览", command=self._browse_model).grid(row=0, column=2, padx=5)

        ttk.Label(frame, text="设备:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.device_var = tk.StringVar(value="cuda")
        ttk.Combobox(frame, textvariable=self.device_var, values=["cuda", "cpu"], width=5, state="readonly").grid(row=1, column=1, sticky=tk.W, padx=5)

    def _create_mcts_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="MCTS设置", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(frame, text="模拟次数:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.simulations_var = tk.IntVar(value=300)
        ttk.Spinbox(frame, from_=20, to=500, textvariable=self.simulations_var, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(frame, text="探索系数(c_puct):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.c_puct_var = tk.DoubleVar(value=1.5)
        c_puct_scale = ttk.Scale(frame, from_=0.0, to=10.0, variable=self.c_puct_var, orient=tk.HORIZONTAL, length=150)
        c_puct_scale.grid(row=1, column=1, padx=5)
        self.c_puct_label = ttk.Label(frame, text="1.5")
        self.c_puct_label.grid(row=1, column=2, padx=5)
        self.c_puct_var.trace('w', lambda *args: self.c_puct_label.configure(text=f"{self.c_puct_var.get():.1f}"))

        ttk.Label(frame, text="温度参数:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.temperature_var = tk.DoubleVar(value=0.6)
        temp_scale = ttk.Scale(frame, from_=0.1, to=2.0, variable=self.temperature_var, orient=tk.HORIZONTAL, length=150)
        temp_scale.grid(row=2, column=1, padx=5)
        self.temperature_label = ttk.Label(frame, text="0.6")
        self.temperature_label.grid(row=2, column=2, padx=5)
        self.temperature_var.trace('w', lambda *args: self.temperature_label.configure(text=f"{self.temperature_var.get():.2f}"))

    def _create_training_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="训练设置", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))

        # 仅训练模式复选框
        self.train_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="仅训练模式",
                        variable=self.train_only_var).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(frame, text="训练局数:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.total_var = tk.IntVar(value=1000)
        ttk.Spinbox(frame, from_=10, to=10000, textvariable=self.total_var, width=15).grid(row=1, column=1, padx=5)

        ttk.Label(frame, text="进程数:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.threads_var = tk.IntVar(value=4)
        ttk.Spinbox(frame, from_=1, to=16, textvariable=self.threads_var, width=10).grid(row=2, column=1, padx=5,
                                                                                         sticky=tk.W)

        ttk.Label(frame, text="批大小:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.batch_size_var = tk.IntVar(value=512)
        ttk.Spinbox(frame, from_=32, to=4096, textvariable=self.batch_size_var, width=10).grid(row=3, column=1, padx=5,
                                                                                               sticky=tk.W)

        ttk.Label(frame, text="模型保存间隔(局):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.save_interval_var = tk.IntVar(value=4)
        ttk.Spinbox(frame, from_=1, to=20, textvariable=self.save_interval_var, width=10).grid(row=4, column=1, padx=5,
                                                                                               sticky=tk.W)

        # 仅训练模式的额外参数
        ttk.Label(frame, text="训练轮数（仅训练模式）:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.train_epochs_var = tk.IntVar(value=10)
        ttk.Spinbox(frame, from_=1, to=1000, textvariable=self.train_epochs_var, width=15).grid(row=5, column=1, padx=5)
        # 在 _create_training_settings 方法中，找到合适的位置插入
        self.exploration_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="探索模式（开局随机放黑子白子）",
                        variable=self.exploration_mode_var).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=5)
    def _create_optimizer_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="优化器设置", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(frame, text="学习率:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.DoubleVar(value=0.0001)
        lr_scale = ttk.Scale(frame, from_=0.00000, to=0.0005, variable=self.lr_var, orient=tk.HORIZONTAL, length=150)
        lr_scale.grid(row=0, column=1, padx=5)
        self.lr_label = ttk.Label(frame, text="0.0001")
        self.lr_label.grid(row=0, column=2, padx=5)
        self.lr_var.trace('w', lambda *args: self.lr_label.configure(text=f"{self.lr_var.get():.6f}"))

        ttk.Label(frame, text="权重衰减:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.weight_decay_var = tk.DoubleVar(value=0.0001)
        wd_scale = ttk.Scale(frame, from_=0.0, to=0.001, variable=self.weight_decay_var, orient=tk.HORIZONTAL, length=150)
        wd_scale.grid(row=1, column=1, padx=5)
        self.wd_label = ttk.Label(frame, text="0.0001")
        self.wd_label.grid(row=1, column=2, padx=5)
        self.weight_decay_var.trace('w', lambda *args: self.wd_label.configure(text=f"{self.weight_decay_var.get():.6f}"))

    def _create_data_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="数据管理", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))

        self.load_data_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="加载已有数据", variable=self.load_data_var).pack(anchor=tk.W)

        self.data_dir_var = tk.StringVar(value="data/")
        ttk.Entry(frame, textvariable=self.data_dir_var, width=25).pack(fill=tk.X, pady=2)

    def _create_buttons(self, parent):
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)

        self.start_btn = ttk.Button(button_frame, text="开始训练", command=self._start_training)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(button_frame, text="停止训练", command=self._stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.apply_btn = ttk.Button(button_frame, text="应用超参数", command=self._apply_hyperparams)
        self.apply_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="保存模型", command=self._save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="加载数据", command=self._load_data).pack(side=tk.LEFT, padx=5)

    def _create_board_area(self, parent):
        board_frame = ttk.LabelFrame(parent, text="对局回放", padding="10")
        board_frame.pack(fill=tk.BOTH, expand=True)

        self.board_size = 15
        self.cell_size = 35
        self.margin = 35
        self.board_width = (self.board_size - 1) * self.cell_size
        self.board_canvas_size = self.board_width + 2 * self.margin

        self.canvas = tk.Canvas(board_frame, width=self.board_canvas_size,
                                 height=self.board_canvas_size, bg='#DCB35C', highlightthickness=0)
        self.canvas.pack(pady=10)
        self.game_board = GameBoard(self.canvas, self.board_size, self.cell_size, self.margin)

        # 手动回放控制按钮
        control_frame = ttk.Frame(board_frame)
        control_frame.pack(pady=5)

        # 上一局、下一局
        self.prev_game_btn = ttk.Button(control_frame, text="◀ 上一局", command=self._prev_game, width=8)
        self.prev_game_btn.pack(side=tk.LEFT, padx=5)

        self.next_game_btn = ttk.Button(control_frame, text="下一局 ▶", command=self._next_game, width=8)
        self.next_game_btn.pack(side=tk.LEFT, padx=5)

        # 分隔线
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # 上一步、下一步（步进）
        self.prev_step_btn = ttk.Button(control_frame, text="◀ 上一步", command=self._prev_step, width=8)
        self.prev_step_btn.pack(side=tk.LEFT, padx=5)

        self.next_step_btn = ttk.Button(control_frame, text="下一步 ▶", command=self._next_step, width=8)
        self.next_step_btn.pack(side=tk.LEFT, padx=5)

        # 步数显示标签
        self.step_label = ttk.Label(control_frame, text="步数: 0/0", font=('微软雅黑', 9))
        self.step_label.pack(side=tk.LEFT, padx=10)

        self.game_info_label = ttk.Label(board_frame, text="", font=('微软雅黑', 10))
        self.game_info_label.pack(pady=5)

    def _create_status_area(self, parent):
        status_frame = ttk.LabelFrame(parent, text="训练状态", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.status_text = tk.Text(status_frame, height=10, font=('Consolas', 9), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.stats_panel = TrainingStatsPanel(status_frame)
        self.stats_panel.create()

    def _draw_empty_board(self):
        self.canvas.delete("all")
        self.game_board.draw_grid()
        self.game_board.draw_stars()

    def _draw_board_from_moves(self, moves, step):
        """
        绘制棋盘到指定步数
        moves: 落子列表 [(row, col, player), ...]
        step: 显示的步数（0表示空棋盘，step <= len(moves)）
        """
        self._draw_empty_board()
        for i in range(step):
            r, c, player = moves[i]
            self.game_board.draw_piece(r, c, player)
        if step > 0:
            last_r, last_c, _ = moves[step - 1]
            self.game_board.highlight_move(last_r, last_c)

        # 更新步数显示
        self.step_label.config(text=f"步数: {step}/{len(moves)}")

    def _display_game(self, game_result):
        """显示指定对局，重置步数为最后一步"""
        if game_result is None:
            self._draw_empty_board()
            self.game_info_label.config(text="")
            self.step_label.config(text="步数: 0/0")
            return

        self.current_game_moves = game_result.moves
        self.current_winner = game_result.winner
        self.current_game_id = game_result.game_id
        self.current_step = len(self.current_game_moves)  # 默认显示最后一步
        self._draw_board_from_moves(self.current_game_moves, self.current_step)

        winner_text = "黑棋胜" if self.current_winner == 1 else "白棋胜" if self.current_winner == -1 else "平局"
        self.game_info_label.config(
            text=f"游戏 #{self.current_game_id} | 结果: {winner_text} | 手数: {len(self.current_game_moves)}"
        )
        self.stats_panel.update_current_game(self.current_game_id)

    def _prev_game(self):
        if self.game_history and self.current_history_index > 0:
            self.current_history_index -= 1
            self._display_game(self.game_history[self.current_history_index])

    def _next_game(self):
        if self.game_history and self.current_history_index < len(self.game_history) - 1:
            self.current_history_index += 1
            self._display_game(self.game_history[self.current_history_index])

    def _prev_step(self):
        """回退一步"""
        if self.current_game_moves and self.current_step > 0:
            self.current_step -= 1
            self._draw_board_from_moves(self.current_game_moves, self.current_step)

    def _next_step(self):
        """前进一步"""
        if self.current_game_moves and self.current_step < len(self.current_game_moves):
            self.current_step += 1
            self._draw_board_from_moves(self.current_game_moves, self.current_step)

    def _browse_model(self):
        filename = filedialog.askopenfilename(title="选择模型文件", filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")])
        if filename:
            self.model_path_var.set(filename)

    def _apply_hyperparams(self):
        """应用超参数（训练中禁止修改）"""
        if self.is_training:
            messagebox.showwarning("警告", "训练进行中，请先停止训练后再修改超参数。")
            return
        if self.trainer:
            self.trainer.update_hyperparameters(
                learning_rate=self.lr_var.get(),
                weight_decay=self.weight_decay_var.get(),
                c_puct=self.c_puct_var.get(),
                num_simulations=self.simulations_var.get(),
            )
            self._log_message(f"[超参数] 已更新: lr={self.lr_var.get():.6f}, c_puct={self.c_puct_var.get():.1f}")

    def _start_training(self):
        if self.is_training:
            return

        self.trainer = SelfPlayTrainer(
            model_path=self.model_path_var.get(),
            board_size=15,
            device=self.device_var.get(),
            num_simulations=self.simulations_var.get(),
            data_dir=self.data_dir_var.get(),
            c_puct=self.c_puct_var.get(),
            learning_rate=self.lr_var.get(),
            weight_decay=self.weight_decay_var.get()
        )

        # 保存初始模型（如果不存在）
        if not os.path.exists(self.model_path_var.get()):
            self.trainer.save_model()
            print(f"[训练] 已保存初始模型到 {self.model_path_var.get()}")

        self.game_history = []
        self.current_history_index = -1
        self.status_text.delete(1.0, tk.END)
        self._log_message("=" * 60)
        self._log_message(f"训练开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        train_only = self.train_only_var.get()
        if train_only:
            self._log_message("[模式] 仅训练模式 - 只从已有数据学习，不生成新对局")
        else:
            self._log_message("[模式] 正常模式 - 自我对弈生成数据并训练")

        self.is_training = True
        self.stop_event.clear()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        if train_only:
            training_thread = threading.Thread(
                target=self._train_only_loop,
                args=(self.total_var.get(), self.batch_size_var.get(),
                      self.save_interval_var.get(), self.load_data_var.get(),
                      self.train_epochs_var.get()),
                daemon=True
            )
            training_thread.start()
        else:
            trainer_params = {
                'board_size': 15,
                'num_simulations': self.simulations_var.get(),
                'c_puct': self.c_puct_var.get(),
                'temperature': self.temperature_var.get(),
                'exploration_mode': self.exploration_mode_var.get(),  # 新增
            }
            model_path = self.model_path_var.get()
            num_processes = self.threads_var.get()
            result_queue = mp.Queue()

            self.training_processes = []
            for i in range(num_processes):
                p = mp.Process(
                    target=worker_process,
                    args=(trainer_params, model_path, result_queue, self.stop_event, 'cuda')
                )
                p.start()
                self.training_processes.append(p)

            training_thread = threading.Thread(
                target=self._training_loop,
                args=(result_queue, self.total_var.get(), self.batch_size_var.get(),
                      self.save_interval_var.get(), self.load_data_var.get()),
                daemon=True
            )
            training_thread.start()

    def _train_only_loop(self, total_epochs, batch_size, save_interval, load_data, train_epochs):
        """仅训练模式循环"""
        try:
            if load_data:
                self._log_message("[数据] 正在加载已有数据...")
                self.trainer.load_training_data()
                data_size = len(self.trainer.data_buffer)
                self._log_message(f"[数据] 已加载 {data_size} 条数据")

                if data_size == 0:
                    self._log_message("[错误] 没有可用的训练数据！请先使用正常模式生成数据或加载已有数据文件。")
                    self.update_queue.put({'type': 'finished'})
                    return

            data_size = len(self.trainer.data_buffer)
            if data_size < batch_size:
                self._log_message(f"[警告] 数据量不足 ({data_size} < {batch_size})，请先使用正常模式生成更多数据")
                self.update_queue.put({'type': 'finished'})
                return

            self._log_message(f"[训练] 开始仅训练模式，共 {total_epochs} 轮，每轮训练 {train_epochs} 步")

            for epoch in range(total_epochs):
                if not self.is_training:
                    self._log_message("[训练] 训练被用户中断")
                    break

                epoch_losses = []

                for step in range(train_epochs):
                    if not self.is_training:
                        break

                    policy_loss, value_loss, entropy_loss, total_loss = self.trainer.train_step(batch_size)

                    if policy_loss is not None:
                        epoch_losses.append((policy_loss, value_loss, entropy_loss, total_loss))
                        self.trainer.train_count += 1
                        self.trainer.lr_scheduler.step()

                        if step % 10 == 0:
                            avg_losses = np.mean(epoch_losses[-10:], axis=0) if epoch_losses else (0, 0, 0, 0)
                            self.update_queue.put({
                                'type': 'train_only_progress',
                                'epoch': epoch + 1,
                                'total_epochs': total_epochs,
                                'step': step + 1,
                                'total_steps': train_epochs,
                                'policy_loss': avg_losses[0],
                                'value_loss': avg_losses[1],
                                'entropy_loss': avg_losses[2],
                                'total_loss': avg_losses[3]
                            })

                if epoch_losses:
                    avg_policy = np.mean([l[0] for l in epoch_losses])
                    avg_value = np.mean([l[1] for l in epoch_losses])
                    avg_entropy = np.mean([l[2] for l in epoch_losses])
                    avg_total = np.mean([l[3] for l in epoch_losses])

                    self.update_queue.put({
                        'type': 'train_only_epoch',
                        'epoch': epoch + 1,
                        'total_epochs': total_epochs,
                        'policy_loss': avg_policy,
                        'value_loss': avg_value,
                        'entropy_loss': avg_entropy,
                        'total_loss': avg_total,
                        'data_size': len(self.trainer.data_buffer),
                        'train_count': self.trainer.train_count
                    })

                    self._log_message(f"[轮次 {epoch + 1}/{total_epochs}] "
                                      f"策略损失: {avg_policy:.4f} | "
                                      f"价值损失: {avg_value:.4f} | "
                                      f"熵损失: {avg_entropy:.4f} | "
                                      f"总损失: {avg_total:.4f}")

                if (epoch + 1) % save_interval == 0:
                    self.trainer.save_model()
                    self.update_queue.put({'type': 'log', 'message': f"[模型] 已保存 (轮次 {epoch + 1})"})

            self.trainer.save_model()
            self._log_message("[完成] 仅训练模式完成")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_queue.put({'type': 'error', 'message': str(e)})
        finally:
            self.update_queue.put({'type': 'finished'})

    def _training_loop(self, result_queue, total_games, batch_size, save_interval, load_data):
        """正常训练循环，每局游戏后立即保存数据"""
        try:
            if load_data:
                self.trainer.load_training_data()

            games_processed = 0

            while games_processed < total_games and self.is_training:
                try:
                    states, policies, players, moves, winner = result_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                games_processed += 1
                game_id = games_processed

                # 添加到数据缓冲区
                self.trainer.data_buffer.add_game(states, policies, players, winner)
                self.trainer.game_count += 1

                # 每局游戏后立即保存数据
                game_data = (states, policies, players, winner)
                self.trainer.save_training_data(game_data)

                # 训练步骤
                policy_loss, value_loss, entropy_loss, total_loss = self.trainer.train_step(batch_size)
                if policy_loss is not None:
                    self.trainer.train_count += 1
                    self.trainer.lr_scheduler.step()

                # 创建游戏结果
                game_result = GameResult(game_id, self.trainer.board_size, moves, winner)

                # 更新 GUI
                self.update_queue.put({
                    'type': 'game',
                    'game_count': game_id,
                    'game_result': game_result,
                    'policy_loss': policy_loss,
                    'value_loss': value_loss,
                    'entropy_loss': entropy_loss,
                    'total_loss': total_loss
                })

                # 定期保存模型
                if games_processed % save_interval == 0:
                    self.trainer.save_model()
                    self.update_queue.put({'type': 'log', 'message': f"[模型] 已保存"})

                # 更新统计显示
                self.update_queue.put({'type': 'stats'})

            # 训练完成，清理
            self.stop_event.set()
            for p in self.training_processes:
                p.join(timeout=2)

            self.trainer.save_model()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_queue.put({'type': 'error', 'message': str(e)})
        finally:
            self.update_queue.put({'type': 'finished'})

    def _stop_training(self):
        """停止训练"""
        self.is_training = False
        self.stop_event.set()
        self._log_message("\n[用户] 正在停止训练...")

    def _save_model(self):
        if self.trainer:
            self.trainer.save_model()
            self._log_message(f"[模型] 已保存")
            messagebox.showinfo("成功", "模型已保存")

    def _load_data(self):
        if self.trainer:
            filename = filedialog.askopenfilename(title="选择训练数据文件", initialdir=self.data_dir_var.get())
            if filename:
                self.trainer.load_training_data(filename)
                self._log_message(f"[数据] 已加载: {filename}")

    def _log_message(self, message):
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)

    def _update_stats(self):
        if self.trainer and hasattr(self.trainer, 'data_buffer'):
            data_size = len(self.trainer.data_buffer)
            game_count = getattr(self.trainer, 'game_count', 0)
            train_count = getattr(self.trainer, 'train_count', 0)
            lr = self.trainer.optimizer.param_groups[0]['lr']
            self.stats_panel.update_stats(data_size, game_count, train_count, lr)

    def _process_queue(self):
        """在主线程中处理消息队列"""
        try:
            while not self.update_queue.empty():
                msg = self.update_queue.get_nowait()
                if msg['type'] == 'game':
                    if msg['game_result'] and msg['game_result'].moves:
                        self._display_game(msg['game_result'])
                    if msg['policy_loss'] is not None:
                        self.stats_panel.update_loss(msg['policy_loss'], msg['value_loss'], msg['entropy_loss'],
                                                     msg['total_loss'])
                    self.game_history.append(msg['game_result'])
                    self.current_history_index = len(self.game_history) - 1
                    winner_text = "黑胜" if msg['game_result'].winner == 1 else "白胜" if msg[
                                                                                              'game_result'].winner == -1 else "平局"
                    self._log_message(
                        f"[游戏 {msg['game_count']}] {winner_text} | 手数: {len(msg['game_result'].moves)}")

                elif msg['type'] == 'train_only_progress':
                    self.stats_panel.update_loss(
                        msg['policy_loss'],
                        msg['value_loss'],
                        msg['entropy_loss'],
                        msg['total_loss']
                    )
                    self._log_message(f"[训练进度] 轮次 {msg['epoch']}/{msg['total_epochs']} | "
                                      f"步骤 {msg['step']}/{msg['total_steps']}")

                elif msg['type'] == 'train_only_epoch':
                    self.stats_panel.update_loss(
                        msg['policy_loss'],
                        msg['value_loss'],
                        msg['entropy_loss'],
                        msg['total_loss']
                    )
                    self.stats_panel.update_stats(
                        msg['data_size'],
                        msg['epoch'],
                        msg['train_count'],
                        self.trainer.optimizer.param_groups[0]['lr'] if self.trainer else 0
                    )

                elif msg['type'] == 'stats':
                    self._update_stats()

                elif msg['type'] == 'log':
                    self._log_message(msg['message'])

                elif msg['type'] == 'error':
                    self._log_message(f"[错误] {msg['message']}")

                elif msg['type'] == 'finished':
                    self._log_message("\n[完成] 训练结束")
                    self.is_training = False
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)

            self._update_stats()
        except Exception as e:
            self._log_message(f"[GUI处理错误] {e}")
        finally:
            self.root.after(100, self._process_queue)


if __name__ == "__main__":
    mp.freeze_support()
    app = TrainingGUI()