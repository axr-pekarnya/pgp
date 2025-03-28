import matplotlib.pyplot as plt

# Чтение данных из первого файла
filename1 = "1.txt"  # Укажите первый файл
filename2 = "2.txt"  # Укажите второй файл

def read_data(filename):
    try:
        with open(filename, "r") as file:
            return [float(line.strip()) for line in file]  # Преобразуем строки в числа
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден.")
        exit()
    except ValueError:
        print(f"Ошибка: файл '{filename}' содержит некорректные данные.")
        exit()

# Читаем данные из обоих файлов
data1 = read_data(filename1)
data2 = read_data(filename2)

# Определяем аргументы от 0 до длины данных
x_values = list(range(len(data1)))  # Используем индексы для оси X

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(x_values, data1, marker='o', linestyle='-', color='b')
plt.plot(x_values, data2, marker='s', linestyle='--', color='r')
plt.xlabel("Разрешение кадра")
plt.ylabel("Время рендеринга")
plt.legend()
plt.grid()
plt.show()
