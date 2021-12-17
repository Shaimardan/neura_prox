# #просто сгененриркуем значения . На практике - Если мы функцию знать не должны, то данные берем из
# # например замерять ,
# import math
# import random
#
# import numpy as np
#
#
# class loader:
#
#     def __init__(self,
#               dimensions=2,
#               trainPercent=85.0):# 85 процентов обучающая выборка 15 тестовая
#         self.__tp=trainPercent#__-приватная . _ -protected . размер обучающего множества
#         self.__tr,self.__ts=self.__loadData(dimensions)
#         # типы разные поэтому подсвечивает.
#         # a,b=c в питоне функция может возвращать несколько значений. Функция возвращщает кортеж. Мы можем присвоить
#         # Лоад дата должна вернуть два значения
#
#     def __loadData(self,dim):
#         data=self.__get2DData() if dim==2 else self.__get3DData()
#         ln=len(data)
#         lnts=int(ln*(1-self.__tp/100.))
#         lntr=ln-lnts
#
#         random.shuffle(data)
#         return sorted(data[:lntr]),sorted(data[lntr:])
#
#
#
#
#     def __get3DData(self):
#         # x = np.array([i / 10 for i in range(-30, 31)])
#         # y = np.array( [-i / 10 for i in range(-30, 31)])
#         x = np.linspace(-5, 5, 100)
#         y = np.linspace(-5, 5, 100)
#         z = np.zeros((len(x), len(y)))
#         arr2D = []
#         for i in range(len(x)):
#             for j in range(len(y)):
#                 z[i][j] = math.cos(x[i] + y[j])
#                 arr2D.append([[x[i], y[j]], [z[i][j]]])
#         return arr2D
#         pass
#
#     def __get2DData(self):
#         return[
#             [
#                  [i/10],
#                  [math.cos(i/10)+random.random()*0.2-0.1]
#             ]
#             for i in range (-60,61)
#
#         ]
#
#     def getTrainInp(self):
#         return[v[0] for v in self.__tr] #возвращает двумерный массив
#
#     def getTrainOut(self):
#         return [v[1] for v in self.__tr]  # возвращает двумерный массив
#
#     def getTestOut(self):
#         return [v[1] for v in self.__ts]  # возвращает двумерный массив
#
#     def getTestInp(self):
#         return [v[1] for v in self.__ts]  # возвращает двумерный массив
import math
import random
import numpy as np

class loader:
    def __init__(self, dimensions = 2, trainPercent = 85.0):
        self.__tp = trainPercent
        self.__tr, self.__ts = self.__loadData(dimensions)

    def __loadData(self, dim):
        data = self.__get2DData() if dim == 2 else self.__get3DData()
        ln = len(data)
        lnts =  int(ln * (1 - self.__tp / 100.))
        lntr = ln - lnts

        random.shuffle(data)
        return sorted(data[:lntr]), sorted(data[lntr:])

    def __get2DData(self):
        return [
            [
                [i / 10],
                [math.cos(i / 10) + random.random() * 0.2 - 0.1]
            ]
            for i in range(-60, 61)
        ]

    def __get3DData(self):
        #x = np.array([i / 10 for i in range(-30, 31)])
        #y = np.array( [-i / 10 for i in range(-30, 31)])
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        z = np.zeros((len(x), len(y)))
        arr2D = []
        for i in range(len(x)):
            for j in range(len(y)):
                z[i][j] = math.cos(x[i] + y[j])
                arr2D.append([[x[i], y[j]], [z[i][j]]])
        return arr2D
        '''return [
            [
                [x[i],  y[i]],
               # [math.cos(x[i] + y[i])  + random.random() * 0.02 - 0.01]
                [x[i]**2 + y[i]**2 + random.random() * 0.02 - 0.01]
            ]
            for i in range(len(x))
        ]'''

    def getTrainInp(self):
         return [ v[0] for v in self.__tr] # 2-мерный список, v = [[i / 10], [[math.cos(i / 10) + random.random() * 0.2 - 0.1]]]

    def getTrainOut(self):
        return [ v[1] for v in self.__tr]
    def getTestInp(self):
        return [ v[0] for v in self.__ts]
    def getTestOut(self):
        return [v[1] for v in self.__ts]