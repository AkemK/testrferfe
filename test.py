import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def create_summary_plots():
    sizes = [10, 100, 200, 500, 1000]
    densities = list(np.arange(0.05, 1.0, 0.1))  # От 0.05 до 0.95 с шагом 0.1
    repeat = 5  # Количество повторений для усреднения

    # Словарь для хранения всех данных
    all_data = {}

    for density in densities:
        density_key = f"density_{density:.1f}"
        all_data[density_key] = {
            'Размер матрицы': sizes,
            'CSR умножение на вектор': [],
            'CSC умножение на вектор': [],
            'Плотное умножение на вектор': [],
            'CSR умножение на скаляр': [],
            'CSC умножение на скаляр': [],
            'Плотное умножение на скаляр': []
        }

        for size in sizes:
            # Генерируем матрицу и вектор один раз для всех тестов
            matrix = generate_sparse_matrix(size, size, density)
            vector = Vector([random.uniform(-1000, 1000) for _ in range(size)])
            scalar = random.uniform(-1000, 1000)

            csr = SparseMatrixCSR(matrix)
            csc = SparseMatrixCSC(matrix)

            # Функция для точного замера времени с повторениями
            def measure_time(func, repeats=repeat):
                times = []
                for _ in range(repeats):
                    start = time.perf_counter()  # Более точный таймер
                    func()
                    end = time.perf_counter()
                    times.append(end - start)
                return np.mean(times) * 1000  # Преобразуем в миллисекунды

            # CSR умножение на вектор
            csr_vector_time = measure_time(lambda: csr * vector)
            all_data[density_key]['CSR умножение на вектор'].append(csr_vector_time)

            # CSC умножение на вектор
            csc_vector_time = measure_time(lambda: csc * vector)
            all_data[density_key]['CSC умножение на вектор'].append(csc_vector_time)

            # Плотное умножение на вектор
            dense_vector_time = measure_time(lambda: matrix * vector)
            all_data[density_key]['Плотное умножение на вектор'].append(dense_vector_time)

            # CSR умножение на скаляр
            csr_scalar_time = measure_time(lambda: csr * scalar)
            all_data[density_key]['CSR умножение на скаляр'].append(csr_scalar_time)

            # CSC умножение на скаляр
            csc_scalar_time = measure_time(lambda: csc * scalar)
            all_data[density_key]['CSC умножение на скаляр'].append(csc_scalar_time)

            # Плотное умножение на скаляр
            dense_scalar_time = measure_time(lambda: matrix * scalar)
            all_data[density_key]['Плотное умножение на скаляр'].append(dense_scalar_time)

    # Создаем отдельные холсты для каждой плотности
    for density in densities:
        density_key = f"density_{density:.1f}"
        data = all_data[density_key]

        # Создаем холст с 4 субграфиками для текущей плотности
        plt.figure(figsize=(16, 12))
        plt.suptitle(f'Анализ производительности матричных операций (Плотность = {density:.1f})',
                     fontsize=16, fontweight='bold')

        # График 1: Сравнение умножения на вектор
        plt.subplot(2, 2, 1)
        plt.plot(data['Размер матрицы'], data['CSR умножение на вектор'],
                 marker='o', linewidth=2, label='CSR', color='blue')
        plt.plot(data['Размер матрицы'], data['CSC умножение на вектор'],
                 marker='s', linewidth=2, label='CSC', color='red')
        plt.plot(data['Размер матрицы'], data['Плотное умножение на вектор'],
                 marker='^', linewidth=2, label='Плотное', color='green')
        plt.xlabel('Размер матрицы')
        plt.ylabel('Время (мс)')
        plt.title('Умножение матрицы на вектор')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # График 2: Сравнение умножения на скаляр
        plt.subplot(2, 2, 2)
        plt.plot(data['Размер матрицы'], data['CSR умножение на скаляр'],
                 marker='o', linewidth=2, label='CSR', color='blue')
        plt.plot(data['Размер матрицы'], data['CSC умножение на скаляр'],
                 marker='s', linewidth=2, label='CSC', color='red')
        plt.plot(data['Размер матрицы'], data['Плотное умножение на скаляр'],
                 marker='^', linewidth=2, label='Плотное', color='green')
        plt.xlabel('Размер матрицы')
        plt.ylabel('Время (мс)')
        plt.title('Умножение матрицы на скаляр')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # График 3: Ускорение относительно плотного формата
        plt.subplot(2, 2, 3)
        speedup_csr = [dense / csr if csr != 0 else 1
                       for dense, csr in zip(data['Плотное умножение на вектор'],
                                             data['CSR умножение на вектор'])]
        speedup_csc = [dense / csc if csc != 0 else 1
                       for dense, csc in zip(data['Плотное умножение на вектор'],
                                             data['CSC умножение на вектор'])]
        plt.plot(data['Размер матрицы'], speedup_csr,
                 marker='s', linewidth=2, label='CSR ускорение', color='blue')
        plt.plot(data['Размер матрицы'], speedup_csc,
                 marker='^', linewidth=2, label='CSC ускорение', color='red')
        plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Нет ускорения')
        plt.xlabel('Размер матрицы')
        plt.ylabel('Коэффициент ускорения')
        plt.title('Ускорение относительно плотного формата')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # График 4: Относительная эффективность CSR vs CSC
        plt.subplot(2, 2, 4)
        ratio_vector = [csr / csc if csc != 0 else 1
                        for csr, csc in zip(data['CSR умножение на вектор'],
                                            data['CSC умножение на вектор'])]
        ratio_scalar = [csr / csc if csc != 0 else 1
                        for csr, csc in zip(data['CSR умножение на скаляр'],
                                            data['CSC умножение на скаляр'])]

        plt.plot(data['Размер матрицы'], ratio_vector,
                 marker='o', linewidth=2, label='Умножение на вектор', color='purple')
        plt.plot(data['Размер матрицы'], ratio_scalar,
                 marker='s', linewidth=2, label='Умножение на скаляр', color='orange')
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Равенство')
        plt.xlabel('Размер матрицы')
        plt.ylabel('CSR время / CSC время')
        plt.title('Относительная эффективность CSR vs CSC')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'summary_density_{density:.1f}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Дополнительный график: Сравнение всех плотностей на одном холсте (опционально)
    create_density_comparison_plot(all_data, sizes, densities)


def create_density_comparison_plot(all_data, sizes, densities):
    """Дополнительный график для сравнения разных плотностей"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Сравнение производительности для разных плотностей', fontsize=16, fontweight='bold')

    # График 1: CSR умножение на вектор для разных плотностей
    ax1 = axes[0, 0]
    for density in densities:
        density_key = f"density_{density:.1f}"
        data = all_data[density_key]
        ax1.plot(data['Размер матрицы'], data['CSR умножение на вектор'],
                 marker='o', linewidth=2, label=f'density={density:.1f}')
    ax1.set_xlabel('Размер матрицы')
    ax1.set_ylabel('Время (мс)')
    ax1.set_title('CSR умножение на вектор')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: CSC умножение на вектор для разных плотностей
    ax2 = axes[0, 1]
    for density in densities:
        density_key = f"density_{density:.1f}"
        data = all_data[density_key]
        ax2.plot(data['Размер матрицы'], data['CSC умножение на вектор'],
                 marker='s', linewidth=2, label=f'density={density:.1f}')
    ax2.set_xlabel('Размер матрицы')
    ax2.set_ylabel('Время (мс)')
    ax2.set_title('CSC умножение на вектор')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График 3: Плотное умножение на вектор для разных плотностей
    ax3 = axes[1, 0]
    for density in densities:
        density_key = f"density_{density:.1f}"
        data = all_data[density_key]
        ax3.plot(data['Размер матрицы'], data['Плотное умножение на вектор'],
                 marker='^', linewidth=2, label=f'density={density:.1f}')
    ax3.set_xlabel('Размер матрицы')
    ax3.set_ylabel('Время (мс)')
    ax3.set_title('Плотное умножение на вектор')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # График 4: Ускорение CSR для разных плотностей
    ax4 = axes[1, 1]
    for density in densities:
        density_key = f"density_{density:.1f}"
        data = all_data[density_key]
        speedup_csr = [dense / csr if csr != 0 else 1
                       for dense, csr in zip(data['Плотное умножение на вектор'],
                                             data['CSR умножение на вектор'])]
        ax4.plot(data['Размер матрицы'], speedup_csr,
                 marker='o', linewidth=2, label=f'density={density:.1f}')
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Нет ускорения')
    ax4.set_xlabel('Размер матрицы')
    ax4.set_ylabel('Коэффициент ускорения')
    ax4.set_title('Ускорение CSR относительно плотного формата')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('density_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
