import numpy as np
import pandas as pn
# from numpy import array


#1D ARRAY

# lst=[0,3,34,23,3445]
# print(lst)
# arr=np.array(lst)
# print(arr)


# arr2=np.arange(10)
# print(arr2)

# arr3=np.arange(1,5.5,0.1)
# print(arr3)

# arr_f=np.array([2 ,4,5,3,2,3],float)
# print(arr_f)


# arr0=np.array([3 ,5 ,5,54,3])
# print(arr0.ndim)  #izmerenie
# print(arr0.shape) #num of elements in this izmer




# 2D ARRAY

# arrd=np.array([[1,2,3,4],[3,4,2,4]])
# print(arrd)
# print(arrd.ndim)
# print(arrd.shape)
# print(arrd.size)



#3D ARRAY

# arr3d=np.array([[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]])
# print(arr3d)

# arr3d1=np.arange(2,21).reshape(2,3,3)
# print(arr3d1)
# print(arr3d1.ndim)
# print(arr3d1.shape)
# print(arr3d1.size)


#arr=np.zeros(5)
# arr=np.zeros((2,3))
# arr=np.zeros((2,5,9))
# arr=np.ones((4,5,3))

# arr=np.full((2,3,4),5)

# arr=np.empty(3)
# print(arr)


# np.linspace

# arr=np.linspace(1,2,1000)
# print(arr)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(8,6))

# x=np.linspace(-4,4,500000)

# y=(x**3)*(-1)

# plt.grid()
# plt.plot(x,y)

# plt.xlabel('x',fontsize=14)
# plt.ylabel('y',fontsize=14)

# plt.show()

# Задания на одномерные и двумерные массивы
# 1. Одномерные массивы
# Создайте одномерный массив из чисел от 1 до 20 с шагом 2.
# arr=np.arange(1,20,2)
# print(arr)




# Создайте массив из 10 нулей и замените последний элемент на 1.
# arr=np.zeros(10)
# arr[9]=1
# print(arr)


# Используя linspace, создайте массив из 5 элементов, равномерно распределённых между 0 и 10.
# arr=np.linspace(1,10,5)
# print(arr)


# 2. Двумерные массивы
# Создайте двумерный массив размером 3x3, состоящий только из единиц.
# arr=np.ones((3,3))
# print(arr)

# С помощью функции arange создайте массив из чисел от 1 до 12 и преобразуйте его в двумерный массив размером 3x4.
# arr=np.arange(1,13).reshape(3,4)
# print(arr)



# Создайте двумерный массив 4x4, где все элементы на главной диагонали равны 5, а остальные элементы равны 0 (подсказка: используйте np.zeros и np.fill_diagonal).
# arr=np.zeros((4,4))
# np.fill_diagonal(arr,5)
# print(arr)


# Задания на свойства массивов
# Создайте массив из чисел от 1 до 9 и выведите его свойства: форму (shape), размер (size) и тип данных (dtype).
# arr=np.arange(0,10).reshape(2,5)
# print(arr.shape)
# print(arr.size)
# print(arr.dtype)
# print(arr.ndim)




# Создайте двумерный массив размером 2x5 из случайных чисел, преобразуйте его в массив 5x2 и проверьте его форму.
# arr=np.random.rand(2,5)
 
# arr.reshape(5,2)

# print(arr)

# Проверьте, сколько памяти занимает массив из 1000 элементов типа int32 и из 1000 элементов типа float64.
# arr=np.ones(100,dtype=np.int32)
# arr1=np.ones(100,dtype=np.float64)
# print(arr.nbytes)
# print(arr1.nbytes)



# Бонусные задания
# Создайте массив из случайных целых чисел от 10 до 50 размером 3x3. Найдите:

# arr=np.random.randint(10,50,size=(3,3))
# print(arr)

# # Минимальный элемент массива.
# a=np.min(arr)
# print(a)

# # Максимальный элемент массива.
# b=np.max(arr)
# print(b)
# # Среднее значение всех элементов массива.
# s=np.sum(arr)
# print(s/9)

# # Создайте двумерный массив размером 5x5, состоящий из случайных чисел от 0 до 1. Найдите сумму всех элементов.
# arr1=np.random.rand(5,5)
# print(arr1)
# print(np.sum(arr1))



# # Создайте массив, содержащий квадраты чисел от 1 до 10. Выведите только те элементы, которые больше 20.
# arr2=np.arange(1,11)**2
# print(arr2)
 
# arrr=arr2[arr2>20]
# print(arrr)




# PANDAS
# series
# date={'a':2,'d':4}
# datee=pn.Series(date)

# print(date)
# print(datee.index)
# # print(datee.values)
# print(datee.dtype)

# dateframe
# data={
#     'names':['Ali','Vali','Shoh'],
#     'ages':[12,23,32],
#     "adress":['usa','tjk','russia']
# }

# fr=pn.DataFrame(data)

# # print(fr.info)
# # print(fr.columns)
# # print(fr.head)
# print(fr.describe)



# NumPy
# 1. Индексация, срезы, доступ к элементам
# У вас есть массив размером 
# 3
# ×
# 3
# 3×3 с числами от 1 до 9.

# Найдите элемент, который находится во 2-й строке и 3-м столбце.
# arr=np.arange(1,10).reshape(3,3)
# print(arr)


# print(arr[1,2])
# # Извлеките первую строку как отдельный массив.
# arr1=np.array(arr[:1,:])
# print(arr1)

# # Извлеките первый столбец как отдельный массив.
# arr2=np.array(arr[:,:1])
# print(arr2)

# У вас есть массив из чисел от 10 до 50 с шагом 5.
# arr=np.arange(10,50,5)
# print(arr)

# # Выберите элементы с 2-го по 4-й (включительно).
# print(arr[1:4])


# # Установите все чётные числа в массиве равными 0.
# for i in range(len(arr)):
#     if arr[i]%2==0:
#         arr[i]=0

# print(arr)
# Дана матрица 
# 4
# ×
# 4
# 4×4, заполненная случайными числами от 1 до 20.
# arr=np.random.randint(1,20 , size=(4,4))

# print(arr)


# # Замените все элементы на главной диагонали на 0.
# np.fill_diagonal(arr,0)
# print(arr)


# # Извлеките третий столбец как отдельный массив.
# arr2=np.array(arr[:,:1])
# print(arr2)


# Pandas
# 1. Доступ к строкам и столбцам (iloc, loc)
# У вас есть таблица с данными о студентах (имя, возраст, оценки).
# st={
#     'name':['Ali','Vali','Mol'],
#     'age':[12,34,32],
#     'mark':['A','B','C']
# }

# dt=pn.DataFrame(st)


# Выведите информацию о втором студенте, используя индексацию.
# print(dt[1:2])


# Найдите всех студентов с оценкой "A".
# df=dt[dt['mark']=='A']
# print(df)
# Измените возраст одного из студентов, используя его имя.
# df=dt[dt['name']=='Mol']
# df['name']='Lol'

# print(df)

# У вас есть таблица с данными о продажах (товар, количество, цена).

# dictt={
#    'product':['banana','kiwi','apple'],
#    'quentity':[22,24,23],
#    'price':[10.,20.,30.],
    
# }

# df=pn.DataFrame(dictt)

# Найдите строку, где товар — "Бананы".
# df1=df[df['product']=='banana']
# print(df1)


# Выведите только два столбца: названия товаров и их цены.
# m=df.loc[:,['product',"price"]]
# print(m)


# Увеличьте цены всех товаров на 10%.
# df['price']=df['price']*1.1
# print(df)

# 2. Просмотр первых/последних строк (head, tail)
# У вас есть таблица с данными о погоде.
# df = pn.DataFrame(np.random.randint(1, 100, size=(20, 4)), columns=["A", "B", "C", "D"])




# Посмотрите первые 5 строк таблицы.
# print(df.head())

# Посмотрите последние 3 строки таблицы.
# print(df.tail(3))


# У вас есть таблица со случайными числами размером 
# 20
# ×
# 4
# 20×4.
# df = pn.DataFrame(np.random.randint(1, 100, size=(20, 4)), columns=["A", "B", "C", "D"])


# # Выведите первые 10 строк таблицы.
# print(df.head(10))
# # Выведите последние 5 строк таблицы.
# print(df.tail())
# # Найдите среднее значение для одного из столбцов.
# print(df['A'].mean())



# # DAY 3
#   Основные операции
# 1.NumPy:
#  1.Арифметические операции с массивами.
#  2.Логические операции и фильтрация.
# 2.Pandas:
#  1.Добавление/удаление столбцов.
#  2.Фильтрация данных с помощью условий.


# 1.NumPy:
#  1.Арифметические операции с массивами.

# a=np.array([1,2,3,4])
# b=np.array([1,2,3,4])

# print(a+b)
# print(a-b)
# print(a*b)
# print(a/b)
# print(a**2)
# print(np.sqrt(a))
# print(np.sin(90))



#  2.Логические операции и фильтрация.

# ar=np.array([10,15,20,25])

# # print(ar>10)
# ar[(ar>10 ) & (ar<25)] =0
# print(ar)



# 2.Pandas:
#  1.Добавление/удаление столбцов.
#  2.Фильтрация данных с помощью условий.

    #  1.Добавление/удаление столбцов.


# data = {
#     "Product": ["Apple", "Banana", "Cherry"],
#     "Price": [1.2, 0.8, 2.5]
# }


# df=pn.DataFrame(data)
# # print(df)
# df['quantity']=[10,20,30]
# # print(df)

# df['total']=df['Price']*df['quantity']

# df['category']='fruit'

# # df = df.drop(['category','Price'],axis=1 )
# # del df['category' ]



#      #  2.Фильтрация данных с помощью условий.
# fd=df[df['total']>15]
# fk=df.loc[df['Price']>1,['quantity']]
# print(fk)



# # TASKS
# 1. Арифметические операции с массивами
# Создайте два массива размером  3×3, заполненных случайными числами от 1 до 10. Выполните следующие операции:
# ar=np.random.randint(1,5, size=(3,3) )
# arr=np.random.randint(1,5, size=(3,3) )

# print(ar)
# print(arr)
# Сложите массивы.
# print(ar+arr)
# Вычтите один массив из другого.
# print(ar-arr)
# # Найдите поэлементное произведение массивов.
# print(arr*ar)
# Разделите элементы одного массива на другой.
# print(arr/ar)
# Создайте массив из чисел от 1 до 10. Возведите все элементы в квадрат.
# print(arr**2)

# Дан массив 
#   [5,10,15,20,25].
# arr=np.array([5,10,15,20,25])
# # Найдите:

# # Синус каждого элемента.
# print(np.sin(arr))

# # Квадратный корень каждого элемента.
# print(np.sqrt(arr))

# Создайте массив из чисел от 1 до 20. Добавьте к каждому элементу 5 и умножьте результат на 2.

# arr=np.array([1,2,3,4,15,19])
# print(arr)
# print((arr+5)/2)

# У вас есть массив цен товаров: 
#   [100,200,300,400,500]. Увеличьте каждую цену на 10%.
# arr=np.array([100,200,300,400,500])
# print(1.1*arr)


# 2. Логические операции и фильтрация
# Создайте массив из чисел от 1 до 20. Найдите:
# arr=np.random.randint(1,21,size=(5))
 
# Все элементы больше 10.
# ar=arr[arr>10]
# print(ar)
# Все элементы, которые делятся на 3.
# ar=arr[arr%3==0]
# print(ar)
# У вас есть массив 
#   [2,4,6,8,10,12,14,16]. Извлеките все чётные элементы.

# arr=np.array([2,4,6,8,10,12,14,16])
# ar=arr[arr%2!=0]
# print(ar)
# Создайте массив из случайных чисел от 1 до 50 размером 
#   5×5.
# arr=np.random.randint(1,51,size=(5,5))
# print(arr)
# Найдите:

# Все элементы больше 25.
# ar=arr[arr>25]
# print(ar)
# Все элементы, которые находятся в диапазоне от 10 до 30.
# ar=arr[(arr>10 ) & (arr<30)]
# print(ar)
# Создайте массив из чисел от 1 до 15. Замените все элементы, которые больше 10, на  0
# arr=np.random.randint(1,16,size=(5))
# print(arr)
# arr[arr>10]=0
# print(arr)


# Дан массив 
#   [15,22,30,12,18,25,8]. 
# arr=np.array([15,22,30,12,18,25,8])
# # Найдите индексы всех элементов, которые меньше 20.
# i=np.where(arr>20)
# print(i)
# У вас есть массив из чисел 
 
#  [3,9,12,15,20].
# arr=np.array([3,9,12,15,20])

# Определите, какие элементы делятся на 3  а какие — нет.
# print(arr[arr%3==0])
# print(arr[arr%3!=0])


# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt


# # Загрузка данных
# students = pd.read_csv('C:/Users/Hakim/Desktop/pandas/students.csv')
# grades = pd.read_csv('grades.csv')

# # Задание 2: Заполнение пропущенных значений
# grades.fillna(grades.mean(), inplace=True)

# # Задание 3: Добавление среднего балла
# grades['AverageGrade'] = grades[['Math', 'Physics', 'CS']].mean(axis=1)

# # Задание 4: Фильтрация студентов по баллу и посещаемости
# top_students = grades[(grades['AverageGrade'] > 80) & (grades['Attendance (%)'] > 85)]

# # Задание 5: Средние значения по предметам
# subject_averages = grades[['Math', 'Physics', 'CS']].mean()
# overall_average = grades['AverageGrade'].mean()

# # Задание 6: Построение графика посещаемости
# plt.bar(grades.index, grades['Attendance (%)'])
# plt.xlabel('Студенты')
# plt.ylabel('Посещаемость (%)')
# plt.title('Посещаемость студентов')
# plt.show()




# # Задание 7: Сохранение результата
# grades.to_csv('result.csv', index=False)

# import numpy as np
# import pandas as pn
# from matplotlib import pyplot as plt
# import skimage.data as data

# camera = data.camera()  # загружаем тестовое изображение камеры
# binary = camera > 127 
# fig, ax = plt.subplots(1, 2, figsize = (8, 4))
 
# # выведем первое изображение и зададим заголовок
# ax[0].imshow(camera,  cmap = 'gray')
# ax[0].set_title('До')
 
# # для ч/б изображений не забудем про параметр cmap = 'gray'
# ax[1].imshow(binary,  cmap = 'gray')
# ax[1].set_title('После')
# plt.show()



print("hello world")