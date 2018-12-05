#!/usr/bin/python

import os

class splitFile:
    def __init__(self, nevent, output_dir):
        self.nevent = int(nevent)
        self.output_dir = output_dir
        try:
            os.mkdir(self.output_dir)
        except:
            print "dir exist"

    def split(self):
        filein = open("ana/tau0p2.txt","r").read()
        sp = '#txyzpxpypzE'

        for n in range(0,self.nevent):
            fileout = open(output_dir+"P%d.dat"%n,"w")
            head=filein.readline().split()
            NParton=int(head[1])
            fileout.write("%d\n"%NParton)
            for m in range(0,NParton):
                d=filein.readline().split()
                s = "%s %s %s %s %s %s %s %s\n"%(d[5],
                    d[2],d[3],d[4],d[10],d[7],d[8],d[9])
                fileout.write(s)
            fileout.close()
        filein.close()




if __name__=="__main__":
    import sys
    if len(sys.argv) == 3:
        output_dir="ana/ampt_b"+sys.argv[1]+"_LHC_PbPb2p76TeV_tlarge%s/"%sys.argv[2]
    sp = splitFile(100, output_dir)
    sp.split()
