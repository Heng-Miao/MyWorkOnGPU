#!/usr/bin/python

from scipy.optimize import fsolve

def GetScreeningMass( crossection=42, mDGuess=0.6 ):
    '''Get mD from crossection in AMPT, mDGuess is 
    the guess value from user.
    crossection is inversely propotional to mD*mD
    crossection = kF / (mD*mD)
    default:   mD=3.2264     , crossection=3  mb
               mD=2.2814     , crossection=6  mb '''
    kF = 3.0*3.2264**2
    f = lambda x: kF/(x*x)-crossection
    return fsolve( f, mDGuess )[0]


if __name__ == '__main__':
    print GetScreeningMass( crossection=70 )
