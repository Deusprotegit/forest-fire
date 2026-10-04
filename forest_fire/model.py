import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera
import random

# -----------------------------
# Параметры модели
# -----------------------------
m = 50          # число строк
n = 50          # число столбцов
t = 200         # число шагов
k = 1           # порог числа горящих соседей для возгорания
f = 0.01        # вероятность удара молнии
p = 0.01        # вероятность роста нового дерева

# -----------------------------
# Подсчёт горящих соседей
# -----------------------------
def count_burning_neighbors(i, j, mat, m, n):
    """
    Считает число горящих соседей (состояние 2) в окрестности Мура 3x3.
    Границы фиксированные: за пределами поля деревьев нет.
    """
    burning = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni = i + di
            nj = j + dj
            if 0 <= ni < m and 0 <= nj < n:
                if mat[ni, nj] == 2:
                    burning += 1
    return burning

# -----------------------------
# Один шаг модели (синхронный)
# -----------------------------
def step(mat, m, n, k, f, p):
    new_mat = mat.copy()
    for i in range(m):
        for j in range(n):
            s = mat[i, j]
            burning = count_burning_neighbors(i, j, mat, m, n)

            if s == 2:
                # Горящее дерево становится пустой клеткой
                new_mat[i, j] = 0
            elif s == 0:
                # Пустая клетка: вырастает дерево с вероятностью p,
                # если в окрестности нет горящих
                if burning == 0 and random.random() < p:
                    new_mat[i, j] = 1
                else:
                    new_mat[i, j] = 0
            elif s == 1:
                # Дерево: молния или распространение огня
                if random.random() < f:
                    new_mat[i, j] = 2
                elif burning >= k:
                    new_mat[i, j] = 2
                else:
                    new_mat[i, j] = 1
    return new_mat

# -----------------------------
# Симуляция
# -----------------------------
def simulate(m, n, t, k, f, p, density, seed=None):
    """
    density — начальная доля деревьев (0..1).
    Возвращает:
        burn_fraction — доля клеток, которые горели хотя бы раз за симуляцию,
        mat — итоговая матрица
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Начальное состояние: случайное распределение деревьев, горящих нет
    mat = (np.random.rand(m, n) < density).astype(int)

    burned_once = np.zeros((m, n), dtype=bool)

    for _ in range(t):
        # фиксируем клетки, которые горят на этом шаге
        burned_once |= (mat == 2)
        mat = step(mat, m, n, k, f, p)

    # учтём последний шаг
    burned_once |= (mat == 2)

    # Доля сгоревших клеток от начального числа деревьев
    initial_trees = np.sum(mat >= 0)  # просто размер поля
    burn_fraction = burned_once.sum() / (m * n)

    return burn_fraction, mat

# -----------------------------
# Анимация одного запуска
# -----------------------------
def animate_one_run(m, n, t, k, f, p, density=0.6):
    random.seed(42)
    np.random.seed(42)

    mat = (np.random.rand(m, n) < density).astype(int)

    fig = plt.figure(figsize=(5, 5))
    camera = Camera(fig)

    for _ in range(t):
        plt.imshow(mat, cmap='viridis', vmin=0, vmax=2)
        plt.title("Лесной пожар: 0 — пусто, 1 — дерево, 2 — огонь")
        camera.snap()
        mat = step(mat, m, n, k, f, p)

    animation = camera.animate()
    plt.close(fig)
    return animation

# -----------------------------
# Исследование зависимости масштаба пожара от плотности
# -----------------------------
def study_density_dependence():
    densities = np.linspace(0.05, 0.95, 19)
    burn_fractions = []

    for d in densities:
        vals = []
        for seed in range(10):  # усреднение по 10 запускам
            bf, _ = simulate(m, n, t, k, f, p, density=d, seed=seed)
            vals.append(bf)
        burn_fractions.append(np.mean(vals))
        print(f"density={d:.2f}, burn_fraction={burn_fractions[-1]:.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(densities, burn_fractions, 'o-')
    plt.xlabel("Начальная плотность леса")
    plt.ylabel("Доля сгоревших клеток")
    plt.title("Зависимость масштаба пожара от плотности леса")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("burn_vs_density.png", dpi=150)
    plt.show()

# -----------------------------
# Запуск
# -----------------------------
if __name__ == "__main__":
    # Анимация одного запуска
    anim = animate_one_run(m, n, t, k, f, p, density=0.6)
    anim.save("forest_fire.gif", fps=10)  # можно сохранить GIF

    # Исследование зависимости
    study_density_dependence()
