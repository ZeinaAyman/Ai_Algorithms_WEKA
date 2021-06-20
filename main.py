# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 06:58:40 2021

@author: Zeina
"""

import KMeans
import KNN
import DT
import generic

def main():
    choose=input("Choose An algorithm. 1)KNN 2)KMeans 3)Decision Tree 4)Generic Algorithm\n")
    choose=int(choose)
    if choose==1:
        KNN.KNNAlgo()
    elif choose==2:
        KMeans.KMeansAlgo()
    elif choose==3:
        DT.DTAlgo()
    elif choose==4:
        best, score = generic.genetic_algorithm(generic.onemax, generic.n_bits, generic.n_iter, generic.n_pop, generic.r_cross, generic.r_mut)
        print('Done!')
        print('f(%s) = %f' % (best, score))
    else:
            print("\n Please choose from the numbers above.")


if __name__=="__main__":
    main()
