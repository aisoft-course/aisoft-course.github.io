import math
import random

# ---------- 激活函数 ----------

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)


# ---------- 神经网络 ----------

class FactorialNN:
    def __init__(self, hidden_size=10):
        self.hidden_size = hidden_size

        # 输入 -> 隐藏
        self.w1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]

        # 隐藏 -> 输出
        self.w2 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b2 = random.uniform(-1, 1)

    def forward(self, x):
        # 隐藏层
        h = []
        for i in range(self.hidden_size):
            z = self.w1[i] * x + self.b1[i]
            h.append(sigmoid(z))

        # 输出层（线性）
        y = sum(self.w2[i] * h[i] for i in range(self.hidden_size)) + self.b2
        return h, y

    def train(self, x, target, lr=0.01):
        h, y = self.forward(x)
        error = y - target

        # 输出层梯度
        dy = error  # 线性输出，导数=1

        # 隐藏层梯度
        dh = [
            dy * self.w2[i] * sigmoid_derivative(h[i])
            for i in range(self.hidden_size)
        ]

        # 更新隐藏 -> 输出
        for i in range(self.hidden_size):
            self.w2[i] -= lr * dy * h[i]
        self.b2 -= lr * dy

        # 更新输入 -> 隐藏
        for i in range(self.hidden_size):
            self.w1[i] -= lr * dh[i] * x
            self.b1[i] -= lr * dh[i]

        return error ** 2


# ---------- 训练数据 ----------

MAX_N = 10

def log_factorial(n):
    return math.log(math.factorial(n))

training_data = [
    (n / MAX_N, log_factorial(n))
    for n in range(1, MAX_N + 1)
]

nn = FactorialNN(hidden_size=10)

# ---------- 训练 ----------

for epoch in range(20000):
    loss = 0
    for x, y in training_data:
        loss += nn.train(x, y)

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ---------- 测试 ----------

print("\nPrediction:")
for n in range(1, MAX_N + 1):
    x = n / MAX_N
    _, y_pred = nn.forward(x)

    fact_pred = math.exp(y_pred)
    print(f"{n}! ≈ {fact_pred:.1f} (true: {math.factorial(n)})")

