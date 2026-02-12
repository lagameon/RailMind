import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy as np
import random

# === 1. 理论参数设置 ===
GRID_SIZE = 4  # 4x4x4 网格
TOTAL_STEPS = 50  # 模拟时长
DECAY_RATE = 0.3  # 疲劳系数
INPUT_STRENGTH = 2.0  # 外部刺激强度
NUM_TRAINS = 3  # 火车数量

# 火车配色：青、品红、黄绿
TRAIN_COLORS = ['cyan', 'magenta', 'lime']
TRAIN_STARTS = [(0, 0, 0), (3, 3, 3), (0, 3, 0)]


def build_diagonal_grid(size):
    """构建支持对角线移动的 3D 网格图。
    每个节点连接所有 26 邻居（6 面邻 + 12 棱邻 + 8 角邻），
    而不仅仅是 nx.grid_graph 的 6 个轴向邻居。
    """
    G = nx.Graph()
    nodes = []
    for x in range(size):
        for y in range(size):
            for z in range(size):
                nodes.append((x, y, z))
    G.add_nodes_from(nodes)

    for node in nodes:
        x, y, z = node
        # 遍历 3x3x3 邻域（排除自身）
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx_, ny_, nz_ = x + dx, y + dy, z + dz
                    if 0 <= nx_ < size and 0 <= ny_ < size and 0 <= nz_ < size:
                        G.add_edge(node, (nx_, ny_, nz_))
    return G


class LogicTrain:
    """单辆逻辑火车：沿能量梯度搜索，带随机探索"""

    def __init__(self, start_pos, color, trail_len=12):
        self.pos = start_pos
        self.color = color
        self.trajectory = [start_pos]
        self.trail_len = trail_len

    def step(self, graph, energy):
        neighbors = list(graph.neighbors(self.pos))
        if not neighbors:
            return

        # 80% 贪婪（追能量最高）+ 20% 随机探索（避免多车扎堆）
        if random.random() < 0.8:
            next_node = max(neighbors, key=lambda n: energy[n])
        else:
            next_node = random.choice(neighbors)

        if energy[next_node] > 0.15:
            self.pos = next_node
            self.trajectory.append(self.pos)
            energy[next_node] *= (1.0 - DECAY_RATE)

        # 工作记忆容量限制
        if len(self.trajectory) > self.trail_len:
            self.trajectory.pop(0)


class ConsciousnessGrid:
    def __init__(self, size):
        self.size = size
        # 使用支持对角线的图
        self.graph = build_diagonal_grid(size)
        self.nodes = list(self.graph.nodes())

        # 初始化能量
        self.energy = {node: np.random.uniform(0.1, 0.5) for node in self.nodes}

        # 多辆逻辑火车
        self.trains = [
            LogicTrain(TRAIN_STARTS[i], TRAIN_COLORS[i])
            for i in range(NUM_TRAINS)
        ]

    def dynamics_step(self, frame):
        """核心演化算法：能量博弈 + 多路径搜索"""

        # A. 模拟外部输入 (Sensory Input) - 每帧随机注入 2 个刺激点
        for _ in range(2):
            stimulus = random.choice(self.nodes)
            self.energy[stimulus] += INPUT_STRENGTH

        # B. 能量扩散与竞争
        new_energy = self.energy.copy()
        for node in self.nodes:
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                leak = self.energy[node] * 0.08
                new_energy[node] -= leak
                dist_energy = leak / len(neighbors)
                for n in neighbors:
                    new_energy[n] += dist_energy
        self.energy = new_energy

        # C. 所有火车同步决策
        for train in self.trains:
            train.step(self.graph, self.energy)


# === 2. 可视化生成 (3D Animation) ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

mind = ConsciousnessGrid(GRID_SIZE)


def update(frame):
    ax.clear()
    ax.set_facecolor('black')
    mind.dynamics_step(frame)

    # 提取坐标和能量
    x, y, z = [], [], []
    colors = []
    sizes = []

    for node in mind.nodes:
        x.append(node[0])
        y.append(node[1])
        z.append(node[2])
        e = np.clip(mind.energy[node], 0, 2.0)
        colors.append(e)
        sizes.append(e * 100)

    # 1. 画出网格节点
    ax.scatter(x, y, z, c=colors, cmap='plasma', s=sizes, alpha=0.6, edgecolors='none')

    # 2. 画出每辆逻辑火车的轨迹
    for train in mind.trains:
        if len(train.trajectory) > 1:
            tx, ty, tz = zip(*train.trajectory)
            # 轨迹渐变透明：尾部暗，头部亮
            n = len(tx)
            for i in range(n - 1):
                alpha = 0.2 + 0.7 * (i / (n - 1))
                lw = 1.5 + 2.0 * (i / (n - 1))
                ax.plot(
                    [tx[i], tx[i + 1]],
                    [ty[i], ty[i + 1]],
                    [tz[i], tz[i + 1]],
                    c=train.color, linewidth=lw, alpha=alpha,
                )
            # 火车头
            ax.scatter(
                [tx[-1]], [ty[-1]], [tz[-1]],
                c=train.color, s=250, marker='*', edgecolors='white', linewidths=0.5,
            )

    # 设置视觉风格
    ax.set_title(
        f"Neural Consciousness  |  {NUM_TRAINS} Logic Trains  |  Frame {frame}",
        color='white', fontsize=13,
    )
    ax.set_xlabel('X (Visual)', color='gray')
    ax.set_ylabel('Y (Auditory)', color='gray')
    ax.set_zlabel('Z (Memory)', color='gray')
    ax.grid(False)
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_zlim(-0.5, GRID_SIZE - 0.5)
    ax.tick_params(colors='gray')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')

    # 缓慢旋转视角，增加动感
    ax.view_init(elev=25, azim=frame * 2.5)


# 生成动画
ani = animation.FuncAnimation(fig, update, frames=TOTAL_STEPS, interval=200)
ani.save('neural_consciousness.gif', writer='pillow', fps=10)
print("Simulation complete. 'neural_consciousness.gif' generated.")
plt.show()
