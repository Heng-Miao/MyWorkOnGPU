       program test_read
       Integer N1,N2
       Real    F1_M

       OPEN(100,file='ana/zpc.dat')
       READ(100,*)N1,N2,F1_M,N2,N2,N2,N2
       READ(100,*)F1_M,F1_M,F1_M,N1,F1_M,F1_M
       ClOSE(100)

       STOP
       END
