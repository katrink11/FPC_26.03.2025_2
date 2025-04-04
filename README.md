# Перемножение матриц при помощи MPI параллельным способом
Программа реализует параллельное перемножение матриц с распределением вычислений между процессами с помощью технологии MPI.
### Входные параметры

| Параметр               | Описание                                   | Значение/Диапазон       |
|------------------------|--------------------------------------------|-------------------------|
| Размер матрицы         | Размер квадратной матрицы (N×N)            | 500×500 (`MATRIX_SIZE`) |
| Диапазон значений      | Минимальное и максимальное значение        | [1, 10] (`MIN_RAND_VALUE`, `MAX_RAND_VALUE`) |
| Количество процессов   | Число MPI-процессов                        | Задается при запуске    |

### Выходные данные

| Параметр                     | Формат вывода                     | Пример значения        |
|------------------------------|-----------------------------------|------------------------|
| Часть матрицы (5×5)          | Первые 5 строк и 5 столбцов       | `[[3, 5, 2, ...], ...]` |
| Время выполнения             | Секунды с плавающей точкой        | 1.245 sec             |
| Размер матрицы               | Формат N×N                        | 500×500               |
| Число процессов              | Целое число                       | 4                     |

### Пояснения ###
Размер матрицы задается константой MATRIX_SIZE в коде

Для запуска с 4 процессами: mpiexec -n 4 ./program

Выходные данные включают срез матрицы для проверки корректности
