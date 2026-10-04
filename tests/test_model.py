import numpy as np

from forest_fire.model import count_burning_neighbors, simulate, step


def burn_out(mat):
    """Крутит шаги без роста и молний, пока на поле есть огонь."""
    m, n = mat.shape
    while (mat == 2).any():
        mat = step(mat, m, n, 1, 0, 0)
    return mat


def test_single_fire_ignites_eight_neighbors():
    # сплошной лес 5x5, горит только центр
    mat = np.ones((5, 5), dtype=int)
    mat[2, 2] = 2
    new = step(mat, 5, 5, 1, 0, 0)
    assert new[2, 2] == 0            # центр сгорел
    assert (new == 2).sum() == 8     # загорелись все 8 соседей (окрестность Мура)


def test_corner_fire_does_not_wrap_around():
    # границы фиксированные: огонь из угла не «перескакивает» на противоположные края
    mat = np.ones((4, 4), dtype=int)
    mat[0, 0] = 2
    assert count_burning_neighbors(3, 3, mat, 4, 4) == 0
    new = step(mat, 4, 4, 1, 0, 0)
    assert new[0, 3] == 1 and new[3, 0] == 1 and new[3, 3] == 1


def test_no_fire_no_change():
    # без огня, роста и молний поле не меняется
    mat = np.array([[1, 0, 1], [0, 1, 0]])
    assert np.array_equal(step(mat, 2, 3, 1, 0, 0), mat)


def test_empty_column_stops_fire():
    # пустой столбец посередине — противопожарный разрыв
    mat = np.ones((3, 5), dtype=int)
    mat[:, 2] = 0
    mat[:, 0] = 2
    final = burn_out(mat)
    assert (final[:, :2] == 0).all()   # слева всё сгорело
    assert (final[:, 3:] == 1).all()   # справа лес цел


def test_full_forest_burns_completely():
    burn_fraction, steps, reached_right, mat = simulate(
        6, 10, 1000, 1, 0, 0, density=1.0, seed=0, ignite_left=True)
    assert burn_fraction == 1.0
    assert reached_right
    assert steps == 10               # огонь проходит по столбцу за шаг: 10 столбцов
    assert (mat == 0).all()


def test_empty_field():
    burn_fraction, steps, reached_right, _ = simulate(
        5, 5, 100, 1, 0, 0, density=0.0, seed=0, ignite_left=True)
    assert (burn_fraction, steps, reached_right) == (0.0, 0, False)


def test_same_seed_same_result():
    a = simulate(20, 20, 400, 1, 0, 0, density=0.5, seed=7, ignite_left=True)
    b = simulate(20, 20, 400, 1, 0, 0, density=0.5, seed=7, ignite_left=True)
    assert a[:3] == b[:3]
    assert np.array_equal(a[3], b[3])
