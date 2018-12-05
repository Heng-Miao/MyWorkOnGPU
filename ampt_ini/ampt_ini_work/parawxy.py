import numpy as np
import os
import math
import cmath
from subprocess import call
import multiprocessing as mp
import threading as td
import re
from random import *

def randn( nseed=11358913111, low=1, high=2E8):
    '''return N random int numbers '''
    result = 0
    #seed(nseed)
    
    result=randint(low, high) 
    return result

def copyfile(eventID = 0, dest='/home/miaoheng/ideal/ampt_ini/ampt_ini_backup/cent/data0'):
    parent = os.getcwd()
    home = '%s/event%s'%(dest, eventID)
    call(['cp', '-r', 'origin/', home])
    os.chdir( home )
    fexec = open('exec','r').read()
    ran = randn()
    fexec = re.sub( r'nseed_byuser=\d+', 'nseed_byuser=%s'%ran, fexec)
    try:
        open('exec', 'w').write(fexec)
    except:
        print 'change exec failed'

    finpu = open('input.ampt', 'r').read()
    ran1 = randn();
    finpu = re.sub( r'8		! random seed for parton cascade',\
	    '%s  ! random seed for parton cascade'%ran1, finpu)

    finpu = re.sub( r'53153523	! random seed for HIJING',\
	    '%s  ! random seed for HIJING'%randn(), finpu)
    open( 'input.ampt' , 'w').write( finpu )
    os.chdir( parent )

def make_event( eventID = 0, dest='/home/miaoheng/ideal/ampt_ini/ampt_ini_backup/cent/data0' ):
    '''Submit ampt job with different random seed'''
    parent = os.getcwd()
    home = '%s/event%s'%(dest,eventID)
    os.chdir( home )
    call('sh exec',shell=True)
    os.chdir( parent )

def collect( eventsID, fin = '../1' ,fou='../data2'):
    ''' Collect partons distribution in different events,
    put them in one folder.
    '''
    try:
        #print fin
        os.chdir( fin )
        if not os.path.exists(fou):
            os.makedirs(fou)
        #os.mkdir(fin)
    
    except:
        print 'dir exsit or mkdir error'
    i = 0
    for ID in xrange(eventsID):
        try:
            print os.getcwd()
            ftau0p2 = open('event%d/ana/tau0p2.txt'%ID, 'r').read()
            amptbnp=open('event%d/ana/ampt.dat'%ID, 'r').read()
            events = ftau0p2.split('#Epxpypztxyz')
            bnaprt = amptbnp.split('\n')
            for j in xrange(len(events)):
                s = events[j].strip()
                bnpt = (bnaprt[j].strip()).split()
                #print ID,bnpt
                #if s != '' :
                if s != '' and len(events)==51:
                    try:
                        nlines = len(s.split('\n'))
                        fout = open('%s/P%d.txt'%(fou,i), 'w')
                        print >>fout, nlines,bnpt[3],int( bnpt[4] )+ int( bnpt[5] )
                        #print >>fout, nlines
                        print >>fout, s 
                        print 'write event%d succeed'%i,bnpt
                        i = i+1
                    except:
                        print 'cant open for write'
                    
        except IOError:
            print 'cant open file event', ID, '/ana/tau0p2.txt'


if __name__ == '__main__':
	import sys

    	njob = 20                      
    	nevent_perjob = 1              
    	total_events = njob*nevent_perjob #total number of events

    	if len(sys.argv) == 2:
      		if sys.argv[1] == '0':
			for i in range(0,total_events):
				copyfile(i)
			pool=mp.Pool(processes=40)
			for i in range(0,total_events):
				#pass
				pool.apply_async(make_event,args=(i,))
			pool.close()
			pool.join()
                else:
                    collect(total_events, fin = './cent/data0', fou= '../data2')
        else:
        	print "Usage: python ParallizeAMPT.py 0   to make event"
        	print "Usage: python ParallizeAMPT.py 1   to change format"

	
