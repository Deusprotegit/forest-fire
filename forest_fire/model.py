import os

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
def simulate(m, n, t, k, f, p, density, seed=None, ignite_left=False):
    """
    density — начальная доля деревьев (0..1).
    t — максимальное число шагов.
    ignite_left — если True, в начале поджигаются все деревья левого столбца.

    Возвращает:
        burn_fraction — доля сгоревших деревьев от начального числа деревьев
                        (корректна при p = 0, когда новые деревья не растут),
        steps — число выполненных шагов (длительность пожара),
        reached_right — True, если огонь хотя бы раз был в правом столбце,
        mat — итоговая матрица
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Начальное состояние: случайное распределение деревьев, горящих нет
    mat = (np.random.rand(m, n) < density).astype(int)

    # Число деревьев считаем сразу, пока лес ещё не изменился
    initial_trees = int((mat == 1).sum())

    if ignite_left:
        left = mat[:, 0]          # срез — это «окно» в mat, а не копия
        left[left == 1] = 2

    burned_once = np.zeros((m, n), dtype=bool)
    reached_right = False
    steps = 0

    for _ in range(t):
        # фиксируем клетки, которые горят на этом шаге
        burning_now = (mat == 2)
        burned_once |= burning_now
        if burning_now[:, -1].any():
            reached_right = True

        # без молний (f = 0) погасший пожар уже не возобновится — дальше считать незачем
        if f == 0 and not burning_now.any():
            break

        mat = step(mat, m, n, k, f, p)
        steps += 1

    # учтём последний шаг
    burned_once |= (mat == 2)
    if (mat[:, -1] == 2).any():
        reached_right = True

    # Доля сгоревших деревьев от начального числа деревьев
    if initial_trees > 0:
        burn_fraction = burned_once.sum() / initial_trees
    else:
        burn_fraction = 0.0

    return burn_fraction, steps, reached_right, mat

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
def study_density_dependence(runs=30, out_dir="results"):
    """
    Чтобы увидеть влияние именно начальной плотности, рост и молнии отключены
    (p = 0, f = 0), а пожар начинается с поджога левого столбца.
    Для каждой плотности делается runs прогонов с разными seed.
    Результаты сохраняются в out_dir: таблица CSV и три графика.
    """
    os.makedirs(out_dir, exist_ok=True)

    # грубая сетка по всему диапазону + мелкая около ожидаемого порога
    # (для окрестности Мура порог перколяции ~0.407)
    densities = np.unique(np.round(np.concatenate([
        np.arange(0.05, 1.00, 0.05),
        np.arange(0.30, 0.52, 0.02),
    ]), 2))

    t_max = m * n   # заведомо больше длительности любого пожара на таком поле

    rows = []
    for d in densities:
        fractions, durations, reached = [], [], []
        for seed in range(runs):
            bf, steps, rr, _ = simulate(m, n, t_max, k, 0, 0, density=d,
                                        seed=seed, ignite_left=True)
            fractions.append(bf)
            durations.append(steps)
            reached.append(rr)
        rows.append([d, np.mean(fractions), np.std(fractions),
                     np.mean(reached), np.mean(durations), np.std(durations)])
        print(f"density={d:.2f}, burn_fraction={rows[-1][1]:.3f}, "
              f"p_reach={rows[-1][3]:.2f}, duration={rows[-1][4]:.1f}")

    rows = np.array(rows)
    np.savetxt(os.path.join(out_dir, "density_study.csv"), rows, delimiter=",",
               header="density,burn_mean,burn_std,p_reach,duration_mean,duration_std",
               comments="", fmt="%.4f")

    d = rows[:, 0]
    p_c = 0.407  # теоретический порог перколяции узлов для окрестности Мура

    plt.figure(figsize=(8, 5))
    plt.errorbar(d, rows[:, 1], yerr=rows[:, 2], fmt='o-', capsize=3)
    plt.axvline(p_c, linestyle='--', color='gray', label=f"теория: {p_c}")
    plt.xlabel("Начальная плотность леса")
    plt.ylabel("Доля сгоревших деревьев")
    plt.title(f"Масштаб пожара от плотности леса (поле {m}x{n}, {runs} прогонов)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "burn_vs_density.png"), dpi=150)

    plt.figure(figsize=(8, 5))
    plt.plot(d, rows[:, 3], 'o-')
    plt.axvline(p_c, linestyle='--', color='gray', label=f"теория: {p_c}")
    plt.xlabel("Начальная плотность леса")
    plt.ylabel("Доля прогонов, где огонь дошёл до правого края")
    plt.title("Вероятность прохождения огня через всё поле")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reach_vs_density.png"), dpi=150)

    plt.figure(figsize=(8, 5))
    plt.errorbar(d, rows[:, 4], yerr=rows[:, 5], fmt='o-', capsize=3)
    plt.axvline(p_c, linestyle='--', color='gray', label=f"теория: {p_c}")
    plt.xlabel("Начальная плотность леса")
    plt.ylabel("Длительность пожара, шагов")
    plt.title("Длительность пожара от плотности леса")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "duration_vs_density.png"), dpi=150)

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
