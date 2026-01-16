"""
Формулы преобразования орбиталей в коде

Этот модуль содержит основные формулы и функции для преобразования
орбитальных параметров, используемых в небесной механике и астрофизике.
"""

import numpy as np


def kepler_to_cartesian(a, e, i, Omega, omega, M, mu=1.0):
    """
    Преобразование кеплеровских элементов в декартовы координаты
    
    Параметры:
    a : float - большая полуось
    e : float - эксцентриситет
    i : float - наклонение
    Omega : float - долгота восходящего узла
    omega : float - аргумент перицентра
    M : float - средняя аномалия
    mu : float - гравитационный параметр системы
    
    Возвращает:
    r_vec, v_vec : tuple - векторы положения и скорости
    """
    
    # Решаем уравнение Кеплера для эксцентрической аномалии E
    def kepler_equation(E, M, e):
        return E - e * np.sin(E) - M
    
    # Используем метод Ньютона для решения уравнения Кеплера
    E = M  # начальное приближение
    tolerance = 1e-10
    max_iterations = 100
    
    for _ in range(max_iterations):
        f = kepler_equation(E, M, e)
        f_prime = 1 - e * np.cos(E)
        E_new = E - f / f_prime
        
        if abs(E_new - E) < tolerance:
            E = E_new
            break
        E = E_new
    
    # Вычисляем истинную аномалию
    tan_E_2 = np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2)
    nu = 2 * np.arctan(tan_E_2)
    
    # Вычисляем радиус-вектор в орбитальной плоскости
    r = a * (1 - e * np.cos(E))
    
    # Компоненты скорости в орбитальной плоскости
    h = np.sqrt(mu * a * (1 - e**2))  # удельный момент импульса
    vr = np.sqrt(mu / a / (1 - e**2)) * e * np.sin(nu)
    vt = h / r  # трансверсальная скорость
    
    # Преобразование из орбитальной системы в инерциальную
    # через матрицу преобразования
    cos_Omega = np.cos(Omega)
    sin_Omega = np.sin(Omega)
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)
    
    # Компоненты радиус-вектора
    x = r * (cos_Omega * cos_omega * cos_nu - sin_Omega * sin_omega * sin_nu)
    x -= r * cos_i * (cos_Omega * sin_omega * cos_nu + sin_Omega * cos_omega * sin_nu)
    y = r * (sin_Omega * cos_omega * cos_nu - cos_Omega * sin_omega * sin_nu)
    y -= r * cos_i * (sin_Omega * sin_omega * cos_nu + cos_Omega * cos_omega * sin_nu)
    z = r * sin_i * (sin_omega * cos_nu + cos_omega * sin_nu)
    
    # Компоненты скорости
    dx = vr * (cos_Omega * cos_omega * cos_nu - sin_Omega * sin_omega * sin_nu)
    dx -= vt * (cos_Omega * cos_omega * sin_nu + sin_Omega * sin_omega * cos_nu)
    dx -= vr * cos_i * (cos_Omega * sin_omega * cos_nu + sin_Omega * cos_omega * sin_nu)
    dx -= vt * cos_i * (-cos_Omega * sin_omega * sin_nu + sin_Omega * cos_omega * cos_nu)
    dy = vr * (sin_Omega * cos_omega * cos_nu - cos_Omega * sin_omega * sin_nu)
    dy -= vt * (sin_Omega * cos_omega * sin_nu + cos_Omega * sin_omega * cos_nu)
    dy -= vr * cos_i * (sin_Omega * sin_omega * cos_nu + cos_Omega * cos_omega * sin_nu)
    dy -= vt * cos_i * (-sin_Omega * sin_omega * sin_nu + cos_Omega * cos_omega * cos_nu)
    dz = vr * sin_i * (sin_omega * cos_nu + cos_omega * sin_nu)
    dz -= vt * sin_i * (-sin_omega * sin_nu + cos_omega * cos_nu)
    
    r_vec = np.array([x, y, z])
    v_vec = np.array([dx, dy, dz])
    
    return r_vec, v_vec


def cartesian_to_kepler(r_vec, v_vec, mu=1.0):
    """
    Преобразование декартовых координат в кеплеровские элементы
    
    Параметры:
    r_vec : array - вектор положения
    v_vec : array - вектор скорости
    mu : float - гравитационный параметр системы
    
    Возвращает:
    a, e, i, Omega, omega, nu : tuple - кеплеровские элементы
    """
    
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    
    # Специфический орбитальный момент импульса
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    # Энергия
    energy = v**2 / 2 - mu / r
    
    # Большая полуось
    a = -mu / (2 * energy)
    
    # Вектор Лапласа-Рунге-Ленца
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = np.linalg.norm(e_vec)
    
    # Наклонение
    i = np.arccos(h_vec[2] / h)
    
    # Узловый вектор
    N_vec = np.cross([0, 0, 1], h_vec)
    N = np.linalg.norm(N_vec)
    
    # Долгота восходящего узла
    if N != 0:
        Omega = np.arccos(N_vec[0] / N)
        if N_vec[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0  # Неопределено для i=0
    
    # Аргумент перицентра
    if N != 0 and e > 0:
        cos_omega = np.dot(N_vec, e_vec) / (N * e)
        omega = np.arccos(np.clip(cos_omega, -1, 1))
        
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    elif e == 0:
        omega = 0  # Неопределено для круговой орбиты
    else:
        omega = 0  # Неопределено
    
    # Истинная аномалия
    if e > 0:
        cos_nu = np.dot(e_vec, r_vec) / (e * r)
        nu = np.arccos(np.clip(cos_nu, -1, 1))
        
        if np.dot(r_vec, v_vec) < 0:
            nu = 2 * np.pi - nu
    else:
        # Для круговой орбиты определяем как угол от узла
        cos_nu = np.dot(N_vec / N, r_vec) / r
        nu = np.arccos(np.clip(cos_nu, -1, 1))
        
        if r_vec[2] < 0:
            nu = 2 * np.pi - nu
    
    return a, e, i, Omega, omega, nu


def orbital_energy(a, mu=1.0):
    """
    Расчет энергии орбиты
    
    Параметры:
    a : float - большая полуось
    mu : float - гравитационный параметр
    
    Возвращает:
    energy : float - удельная энергия орбиты
    """
    return -mu / (2 * a)


def orbital_period(a, mu=1.0):
    """
    Расчет периода орбиты по третьему закону Кеплера
    
    Параметры:
    a : float - большая полуось
    mu : float - гравитационный параметр
    
    Возвращает:
    T : float - период орбиты
    """
    return 2 * np.pi * np.sqrt(a**3 / mu)


def vis_viva_velocity(r, a, mu=1.0):
    """
    Уравнение вив-вива для расчета скорости
    
    Параметры:
    r : float - расстояние от центрального тела
    a : float - большая полуось
    mu : float - гравитационный параметр
    
    Возвращает:
    v : float - скорость
    """
    return np.sqrt(mu * (2/r - 1/a))


def eccentric_anomaly_to_true_anomaly(E, e):
    """
    Преобразование эксцентрической аномалии в истинную аномалию
    
    Параметры:
    E : float - эксцентрическая аномалия
    e : float - эксцентриситет
    
    Возвращает:
    nu : float - истинная аномалия
    """
    tan_nu_2 = np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2)
    nu = 2 * np.arctan(tan_nu_2)
    return nu


def true_anomaly_to_eccentric_anomaly(nu, e):
    """
    Преобразование истинной аномалии в эксцентрическую аномалию
    
    Параметры:
    nu : float - истинная аномалия
    e : float - эксцентриситет
    
    Возвращает:
    E : float - эксцентрическая аномалия
    """
    tan_E_2 = np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2)
    E = 2 * np.arctan(tan_E_2)
    return E


def mean_anomaly_to_eccentric_anomaly(M, e):
    """
    Решение уравнения Кеплера: M = E - e*sin(E)
    для получения эксцентрической аномалии E из средней аномалии M
    
    Параметры:
    M : float - средняя аномалия
    e : float - эксцентриситет
    
    Возвращает:
    E : float - эксцентрическая аномалия
    """
    # Метод Ньютона-Рафсона
    E = M  # начальное приближение
    tolerance = 1e-10
    max_iterations = 100
    
    for _ in range(max_iterations):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        E_new = E - f / f_prime
        
        if abs(E_new - E) < tolerance:
            return E_new
        E = E_new
    
    return E


def orbital_state_vectors_to_elements(r_vec, v_vec, mu=1.0):
    """
    Полное преобразование векторов состояния в кеплеровские элементы
    
    Параметры:
    r_vec : array - вектор положения
    v_vec : array - вектор скорости
    mu : float - гравитационный параметр
    
    Возвращает:
    elements : dict - словарь с кеплеровскими элементами
    """
    a, e, i, Omega, omega, nu = cartesian_to_kepler(r_vec, v_vec, mu)
    
    # Также вычисляем среднюю и эксцентрическую аномалии
    E = true_anomaly_to_eccentric_anomaly(nu, e)
    M = E - e * np.sin(E)
    
    return {
        'semi_major_axis': a,
        'eccentricity': e,
        'inclination': i,
        'longitude_of_ascending_node': Omega,
        'argument_of_periapsis': omega,
        'true_anomaly': nu,
        'eccentric_anomaly': E,
        'mean_anomaly': M
    }


# Пример использования
if __name__ == "__main__":
    print("Пример преобразования кеплеровских элементов в декартовы координаты:")
    
    # Орбитальные элементы Земли (приблизительно)
    a = 1.0          # астрономические единицы
    e = 0.0167       # эксцентриситет
    i = 0.0          # наклонение (для простоты)
    Omega = 0.0      # долгота восходящего узла
    omega = 0.0      # аргумент перицентра
    M = 0.1          # средняя аномалия
    
    r_vec, v_vec = kepler_to_cartesian(a, e, i, Omega, omega, M)
    print(f"Положение: {r_vec}")
    print(f"Скорость: {v_vec}")
    
    # Обратное преобразование
    elements = orbital_state_vectors_to_elements(r_vec, v_vec)
    print("\nОбратно вычисленные элементы:")
    for key, value in elements.items():
        print(f"{key}: {value}")