#!/usr/bin/env python
#coding:utf-8
"""
  Author:  weichen --<cw1987@mails.ccnu.edu.cn>
  Purpose: copy AMPT events in specified centrality
  Created: 08/01/2016
"""

import os 
import sys
import numpy as np 
from glob import glob
from subprocess import call
    
#----------------------------------------------------------------------
def get_centlimit(fpath,cent='0_10'):
    '''to get the low and high limit in given centrality according to multiplicity file
    
    Args:
        fpath: the file path which includes multiplicity limit in different centrality
        cent:  given centrality
        
    Returns: high and low multiciplicity limit at given centrality 
    '''
    c1,c2 = [i+'%' for i in cent.split('_')]
    with open(fpath) as f:
        d = dict(line.split() for line in f.readlines())
        d['100%']=0
    return (int(d[c1]),int(d[c2]))

#----------------------------------------------------------------------
def CopyCentevent(ampt_dir,dest_dir,high_mul,low_mul,nchoose =200):
    '''copy the ampt events in specified centrality into the corresponding folder
    
    Args:
        ampt_dir: the path of source files about ampt events
        dest_dir: the path of the folder which stored the ampt events in specified centrality
        high_mul: the high limit of multiplicity in specified centrality
        low_mul: the low limit of multiplicity in specified centrality
        nchoose: the number of the events in specified centrality
    '''
    nevent = len(glob(os.path.join(ampt_dir,'P*.txt')))
    nstart = 0    
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    #print nchoose
    eventids = []
    for i in xrange(nevent):
        try:
            partons_data = os.path.join(ampt_dir,'P%s.txt'%i)                              
            with open(partons_data, 'r') as f:
                mul = (f.readline().strip()).split()
                if int(mul[0]) < high_mul and int(mul[0]) > low_mul :
                    #call(['cp', partons_data, dest_dir])
                    mul.append(i)                                                    
                    eventids.append(mul)
                    print 'eventid=', i, ' nparton=', mul
                    nstart +=1
                    #print nstart
            if nstart >=nchoose:
                break        
        except:
            continue
    #print eventids      
    with open(os.path.join(dest_dir,'eventids.txt'),'w') as f:
        for i in xrange(len(eventids)):
        	print>>f,int(eventids[i][3]),float(eventids[i][1]),float(eventids[i][2]),float(eventids[i][0])    
    

if __name__ == '__main__':
    if len(sys.argv)!=2:
        print 'please use:python CopyEvents.py *_*'
    else:
        fpath = '/home/miaoheng/ideal/ampt_ini/ampt_ini_backup/centrality_nparton_au2001.dat'
        cent = sys.argv[1]
        high_mul,low_mul= get_centlimit(fpath,cent)
        
        lgdir = '/home/miaoheng/ideal/ampt_ini/ampt_ini_backup/cent/data2'    
        dest_dir = '/home/miaoheng/ideal/ampt_ini/ampt_ini_backup/cent/data3/{}'.format(cent)
        CopyCentevent(lgdir, dest_dir, high_mul, low_mul,nchoose =200)
