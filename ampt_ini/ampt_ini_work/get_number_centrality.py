from subprocess import call
import os
import numpy as np
from glob import glob

def get_list_mul(dest_dir):
    Lmul = []
    Lmul1= []
    Lb=[]
    Lnpart=[]
    flist = glob(os.path.join(dest_dir,'P*.txt'))
    print len(flist)
    #for i in xrange(len(flist)):
    i=0
    for fpath in flist:
        #fname = 'P{}.txt'.format(i)
        #with open(os.path.join(dest_dir,fname),'r') as f:
        with open(fpath,'r') as f:
            mul_tem = f.readline().strip()
            
            mul=(mul_tem).split()
            Lmul.append( int(mul[0]) )
            Lmul1.append( int(mul[0]) )
            Lb.append( float(mul[1]) )
            Lnpart.append( int(mul[2]) )
            print i , mul_tem,fpath
            i=i+1
    Lmul.sort(reverse=True)

    with open('multiplicity.txt','w') as f:
        for mul in Lmul:
            f.write('{}\n'.format(mul))
    np.savetxt('mulbnpart.dat',np.array( zip(Lmul1[:],Lb[:],Lnpart[:]) ))

def get_limit_centrality(fname='multiplicity.txt',cent=10):
    data = np.loadtxt(fname, dtype=int)
    nindex = int(len(data)*cent/100)

    #print nindex                                                                    
    cent_low_limit = np.loadtxt(fname, dtype=int)[nindex]
    #cent_low_limit = np.loadtxt(fname, dtype=int)
    #print cent_low_limit
    return cent_low_limit
    
if __name__ == '__main__': 
    dest_dir = '/home/miaoheng/ideal/ampt_ini/ampt_ini_backup/cent/data2'
    get_list_mul(dest_dir)                                                         
    with open('centrality_nparton_au2001.dat', mode='w') as f:
        for i in range(0,100,1):
            cent_low_limit = get_limit_centrality(cent=i)
            print>>f, '{}%'.format(i), cent_low_limit
