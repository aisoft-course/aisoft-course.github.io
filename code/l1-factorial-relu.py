import math
import random

# ============================================================
# 1. ReLU 激活函数及其导数
# ============================================================

def relu(x):
    """
    ReLU 激活函数
    relu(x) = max(0, x)
    """
    return max(0.0, x)


def relu_derivative(x):
    """
    ReLU 的导数
    x > 0 时导数为 1
    x <= 0 时导数为 0
    注意：这里的 x 是激活前的线性输入 z
    """
    return 1.0 if x > 0 else 0.0


# ============================================================
# 2. 神经网络定义
# ============================================================

class FactorialNNReLU:
    """
    使用 ReLU 作为隐藏层激活函数的前馈神经网络
    用于近似预测 log(n!)
    """

    def __init__(self, hidden_size=10):
        self.hidden_size = hidden_size

        # 输入 -> 隐藏层
        self.w1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]

        # 隐藏层 -> 输出层
        self.w2 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b2 = random.uniform(-1, 1)

    # ========================================================
    # 3. 前向传播
    # ========================================================

    def forward(self, x):
        """
        前向传播
        返回：
        - z：隐藏层线性输入（用于 ReLU 导数）
        - h：隐藏层激活输出
        - y：输出层结果
        """
        z = []
        h = []

        for i in range(self.hidden_size):
            zi = self.w1[i] * x + self.b1[i]
            z.append(zi)
            h.append(relu(zi))

        # 输出层（线性）
        y = sum(self.w2[i] * h[i] for i in range(self.hidden_size)) + self.b2
        return z, h, y

    # ========================================================
    # 4. 训练（反向传播 + 梯度下降）
    # ========================================================

    def train(self, x, target, lr=0.001):
        # ---------- 前向传播 ----------
        z, h, y = self.forward(x)

        # ---------- 误差 ----------
        error = y - target

        # ---------- 输出层梯度 ----------
        dy = error  # 线性输出

        # ---------- 隐藏层梯度 ----------
        dh = []
        for i in range(self.hidden_size):
            grad = dy * self.w2[i] * relu_derivative(z[i])
            dh.append(grad)

        # ---------- 更新隐藏层 -> 输出层 ----------
        for i in range(self.hidden_size):
            self.w2[i] -= lr * dy * h[i]
        self.b2 -= lr * dy

        # ---------- 更新输入层 -> 隐藏层 ----------
        for i in range(self.hidden_size):
            self.w1[i] -= lr * dh[i] * x
            self.b1[i] -= lr * dh[i]

        return error ** 2


# ============================================================
# 5. 构造训练数据
# ============================================================

MAX_N = 10

def log_factorial(n):
    return math.log(math.factorial(n))

training_data = [
    (n / MAX_N, log_factorial(n))
    for n in range(1, MAX_N + 1)
]


# ============================================================
# 6. 训练网络
# ============================================================

nn = FactorialNNReLU(hidden_size=10)

for epoch in range(20000):
    total_loss = 0.0
    for x, y in training_data:
        total_loss += nn.train(x, y)

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss = {total_loss:.4f}")


# ============================================================
# 7. 测试结果
# ============================================================

print("\nPrediction Results (ReLU):")
for n in range(1, MAX_N + 1):
    x = n / MAX_N
    _, _, y_pred = nn.forward(x)
    fact_pred = math.exp(y_pred)
    print(f"{n}! ≈ {fact_pred:.1f} (true: {math.factorial(n)})")

