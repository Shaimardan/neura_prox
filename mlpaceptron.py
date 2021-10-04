
import loader
import numpy as np


class MLP:
    __a = 1
    __b = 1
    def __init__(self,
                 ld:loader.loader,
                 neuronNum: tuple=(4,3)):#нейронная сет с 2 скрытами слоями на первом 4 нейрона на втором 3
         #количество слоев в сети
         self.__layers = len(neuronNum)+2
         #количество нейронов на каждом слое
         inp = ld.getTrainInp()
         out = ld.getTrainOut()
         nN = [len(inp[0]), len(out[0])]
         self.__nN = np.insert(nN, 1, neuronNum)#[1,2]
         self.__nN = np.insert(nN, 1, neuronNum)#[1,4,3,1]
         self.__inp=np.array(inp)
         self.__out=np.array(out)

         self.__tst_inp=np.array(ld.getTrainInp())
         self.__tst_out=np.array(ld.getTestOut())

         self.__w= [
             np.random(
                 self.__nN[i]+1,
                 self.__nN[i+1]+
                 (0 if i == self.__layers-2 else 1)
             )for i in range(self.__layers-1)
         ]
    def nonLineAct(self,x):
        return np.array(self.__a * np.tanh(self.__b * x))

