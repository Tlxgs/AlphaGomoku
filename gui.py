
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
from game import GomokuGame
from mcts import MCTS
from ui_utils import GameBoard


class GomokuGUI:
    def __init__(self, board_size=15, device='cpu'):
        self.board_size = board_size
        self.game = GomokuGame(board_size)
        self.mcts = MCTS(c_puct=1.5, num_simulations=100, temperature=0.0,
                         onnx_path='model.onnx', device='cpu')

        self.cell_size, self.margin = 50, 50
        canvas_size = (board_size - 1) * self.cell_size + 2 * self.margin

        self.root = tk.Tk()
        self.root.title("五子棋 AI")
        self.root.resizable(False, False)

        # ═══════════ 左侧面板（全部 grid） ═══════════
        ctrl = ttk.LabelFrame(self.root, text="设置", padding=8)
        ctrl.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)

        self.sim_var = tk.IntVar(value=100)
        self.cpuct_var = tk.DoubleVar(value=1.5)
        self.temp_var = tk.DoubleVar(value=0.0)
        self.color_var = tk.StringVar(value="黑棋 (先手)")
        self.mode_var = tk.StringVar(value="对弈棋盘")

        params = [
            ("模拟次数:", self.sim_var, 10, 500, 1),
            ("c_puct:",   self.cpuct_var, 0.0, 10.0, 0.1),
            ("温度:",     self.temp_var, 0.0, 2.0, 0.1),
        ]
        for i, (txt, var, lo, hi, inc) in enumerate(params):
            ttk.Label(ctrl, text=txt).grid(row=i, column=0, sticky="w", pady=3)
            ttk.Spinbox(ctrl, from_=lo, to=hi, increment=inc,
                        textvariable=var, width=8).grid(row=i, column=1, pady=3)

        row = len(params)
        ttk.Label(ctrl, text="执子:").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(ctrl, textvariable=self.color_var,
                     values=["黑棋 (先手)", "白棋 (后手)"],
                     state="readonly", width=12).grid(row=row, column=1, pady=3)

        row += 1
        ttk.Label(ctrl, text="显示:").grid(row=row, column=0, sticky="w", pady=3)
        mode_cb = ttk.Combobox(ctrl, textvariable=self.mode_var,
                               values=["对弈棋盘", "MCTS统计", "策略网络概率"],
                               state="readonly", width=12)
        mode_cb.grid(row=row, column=1, pady=3)
        mode_cb.bind("<<ComboboxSelected>>", lambda _: self._draw())

        row += 1
        ttk.Button(ctrl, text="应用参数", command=self._apply_params) \
            .grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        ttk.Button(ctrl, text="重置游戏", command=self._reset) \
            .grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1
        ttk.Button(ctrl, text="退出", command=self.root.quit) \
            .grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)

        # ═══════════ 胜率条（水平，原风格） ═══════════
        row += 1
        ttk.Separator(ctrl, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1
        wr_frame = ttk.LabelFrame(ctrl, text="当前胜率", padding=5)
        wr_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        # wr_frame 内部用 pack（独立容器，不冲突）
        self.win_canvas = tk.Canvas(wr_frame, width=200, height=30,
                                    bg='white', highlightthickness=0)
        self.win_canvas.pack(pady=(0, 2))
        self.win_label = ttk.Label(wr_frame, text="黑棋 50% | 白棋 50%",
                                   font=('微软雅黑', 9))
        self.win_label.pack()

        # ═══════════ 棋盘 ═══════════
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size,
                                bg='#DCB35C', highlightthickness=0)
        self.canvas.grid(row=0, column=1, padx=5, pady=10)
        self.board = GameBoard(self.canvas, board_size, self.cell_size, self.margin)
        self.canvas.bind("<Button-1>", self._on_click)

        # ═══════════ 搜索状态 ═══════════
        self._search_gen = 0       # 代数计数器：递增使旧线程自然退出
        self.search_thread = None

        self._init_mcts()
        self._draw()
        self._start_search()
        self.root.mainloop()

    # ────────────── MCTS 控制 ──────────────

    def _init_mcts(self):
        self.mcts.root = None
        self.mcts.init_root(self.game)

    def _is_human_turn(self):
        return (self.game.current_player == 1) == \
               (self.color_var.get() == "黑棋 (先手)")

    # ────────────── 绘制 ──────────────

    def _draw(self):
        """统一绘制：棋盘 + 统计 + 胜率条"""
        self.canvas.delete("all")
        self.board.draw_grid()
        self.board.draw_stars()
        mode = self.mode_var.get()
        if mode == "对弈棋盘":
            self._draw_pieces()
        else:
            self._draw_stats(mode == "策略网络概率")
        self._draw_winrate()

    def _draw_pieces(self):
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = self.game.board[r, c]
                if p:
                    self.board.draw_piece(r, c, p)
        if self.game.last_move:
            self.board.highlight_move(*self.game.last_move)

    def _draw_stats(self, is_policy):
        root = self.mcts.root
        if not root or not root.children:
            return
        # 半透明已有棋子
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = self.game.board[r, c]
                if p:
                    x = self.margin + c * self.cell_size
                    y = self.margin + r * self.cell_size
                    rad = self.cell_size // 2 - 2
                    self.canvas.create_oval(
                        x - rad, y - rad, x + rad, y + rad,
                        fill='gray' if p == 1 else 'lightgray',
                        outline='gray', stipple='gray50')
        # 一次遍历收集统计，预计算 max
        stats = []
        for m, ch in root.children.items():
            r, c = m
            if self.game.board[r, c] != 0:
                continue
            vis = ch.visit_count
            wr = (1 - ch.get_value()) / 2 if vis > 0 else 0.5
            stats.append((m, vis, wr, ch.prior_prob))
        if not stats:
            return
        max_val = max(s[3] if is_policy else s[2] for s in stats)
        best = max(stats, key=lambda s: s[3] if is_policy else s[1])[0]
        for m, vis, wr, prob in stats:
            if is_policy:
                self.board.draw_stat_circle(m[0], m[1], prob, max_val,
                                            f"{prob * 100:.1f}%")
            elif vis:
                self.board.draw_stat_circle(m[0], m[1], wr, max_val,
                                            f"{vis}\n{wr * 100:.0f}%")
        self.board.draw_best_circle(*best)

    def _draw_winrate(self):
        """水平胜率条（与原代码风格一致）"""
        self.win_canvas.delete("all")
        root = self.mcts.root
        if not root or not root.children:
            self.win_label.config(text="黑棋 --% | 白棋 --%")
            return
        ch = root.children
        total = sum(c.visit_count for c in ch.values())
        if total == 0:
            self.win_label.config(text="黑棋 --% | 白棋 --%")
            return
        wv = sum(c.get_value() * c.visit_count for c in ch.values()) / total
        # 转换到黑棋视角
        if self.game.current_player == 1:
            black_win = (-wv + 1) / 2
        else:
            black_win = (wv + 1) / 2
        white_win = 1 - black_win
        W, H = 200, 30
        bw = int(W * black_win)
        if bw > 0:
            self.win_canvas.create_rectangle(0, 0, bw, H, fill='black', outline='')
        if bw < W:
            self.win_canvas.create_rectangle(bw, 0, W, H, fill='white', outline='')
        self.win_canvas.create_rectangle(0, 0, W, H, outline='gray')
        self.win_label.config(
            text=f"黑棋 {black_win * 100:.1f}% | 白棋 {white_win * 100:.1f}%")

    # ────────────── 交互 ──────────────

    def _on_click(self, event):
        if self.game.game_over or not self._is_human_turn():
            return
        pos = self.board.coord_to_index(event.x, event.y)
        if pos and self.game.is_legal_move(*pos):
            self._do_move(*pos)

    def _do_move(self, r, c):
        """执行落子（人类或AI）并重启搜索"""
        if self.game.game_over or not self.game.is_legal_move(r, c):
            return
        self.game.make_move(r, c)
        self.mcts.update_root(self.game, (r, c))
        self._draw()
        if self.game.game_over:
            self._search_gen += 1          # 终止后台搜索
            msg = "平局！" if self.game.winner == 0 else \
                  f"{'黑' if self.game.winner == 1 else '白'}棋获胜！"
            messagebox.showinfo("游戏结束", msg)
        else:
            self._start_search()           # 以新根节点重启搜索

    # ────────────── 后台持续搜索 ──────────────

    def _start_search(self):
        """启动后台搜索线程。
        人类思考时持续搜索并刷新显示；
        AI回合搜索到目标次数后自动落子。
        通过 _search_gen 代数计数器安全终止旧线程，无需 Lock。
        """
        if self.game.game_over:
            return
        self._search_gen += 1              # 使旧线程在下次循环退出
        gen = self._search_gen
        self.search_thread = threading.Thread(
            target=self._search_worker, args=(gen,), daemon=True)
        self.search_thread.start()

    def _search_worker(self, my_gen):
        """后台搜索主循环（while 持续运行）"""
        n = self.mcts.num_simulations
        sim_count = 0

        while my_gen == self._search_gen and not self.game.game_over:
            self.mcts._simulate(self.game.copy())
            sim_count += 1

            # 每 20 次刷新界面（人类回合时能看到实时统计/胜率变化）
            if sim_count % 20 == 0:
                self.root.after(0, self._draw)

            # ── AI 回合：达到目标模拟次数后落子 ──
            if (my_gen == self._search_gen
                    and not self.game.game_over
                    and not self._is_human_turn()
                    and sim_count >= n):
                root = self.mcts.root
                if root and root.children:
                    best = self._select_move(root)
                    # 回到主线程执行落子
                    self.root.after(0, lambda b=best: self._do_move(*b))
                return                      # 当前线程使命完成，退出

    def _select_move(self, root):
        """根据温度参数选择落子"""
        moves = list(root.children.keys())
        visits = np.array([root.children[m].visit_count for m in moves])
        temp = self.mcts.temperature
        if temp <= 0.1:
            return moves[np.argmax(visits)]
        probs = visits ** (1.0 / temp)
        probs /= probs.sum()
        return moves[np.random.choice(len(moves), p=probs)]

    # ────────────── 按钮回调 ──────────────

    def _apply_params(self):
        self._search_gen += 1              # 终止旧搜索
        self.mcts.num_simulations = self.sim_var.get()
        self.mcts.c_puct = self.cpuct_var.get()
        self.mcts.temperature = self.temp_var.get()
        self._init_mcts()
        self._draw()
        self._start_search()

    def _reset(self):
        self._search_gen += 1
        self.game.reset()
        self._init_mcts()
        self._draw()
        self._start_search()


if __name__ == "__main__":
    GomokuGUI()
