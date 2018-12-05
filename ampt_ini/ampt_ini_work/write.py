import numpy as np
import h5py
import os

#a=np.loadtxt('200_0_5/eventids.txt')
path = '/home/miaoheng/ideal/ampt_ini/ampt_ini_backup/cent/data3'
os.chdir(path)
cent = ['0_5','5_10','10_20','20_30','30_40','40_50','50_60']
eventlist = []
with h5py.File('pbpb5020.h5','w') as f:
    for centid in cent:
        infor = np.loadtxt('{cent}/eventids.txt'.format(cent=centid))
        f.create_dataset('cent/{cent}/'.format(cent=centid),data=infor)
        events,b,npart,nparton = np.loadtxt('{cent}/eventids.txt'.format(cent=centid),unpack=True)
        eventlist.extend(events)
		
    datalist = np.unique( np.array(eventlist).flatten() )
    print datalist, len(datalist), len(np.unique(datalist) )


    i=0
    for eventid in datalist:
        data = np.loadtxt('../data2/P{eventid}.txt'.format(cent=centid,eventid=int(eventid)),skiprows=1 )
        f.create_dataset('event%d'%eventid,data=data)
        
        print "Event%d  done"%(eventid),i, len(datalist), len(np.unique(datalist) )
        i=i+1
