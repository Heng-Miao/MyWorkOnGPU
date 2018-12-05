#!/usr/bin/python

import sys
import re

t_large = '0.4'
if len(sys.argv) == 2:
    t_large = sys.argv[1]

try:
    zpcf = open("zpc.f",'r').read()
    zpcf_1 = re.sub(r'tlarge=\d+\.\d+d0', 'tlarge=%sd0'%t_large, zpcf)
    try:
        open('zpc.f', 'w').write(zpcf_1)
    except IOError:
        print "Can't write to zpc.f"
except IOError:
    print "Can't open zpc.f"


