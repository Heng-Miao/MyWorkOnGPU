      program ridge
C.....C12 calculation variables..........................
C..... AN=<N>, AN1N2=<N1N2>, AN1AN2=<N1><N2>.............
      INTEGER   NETA, NPHI, NTOTAL, ITEMP, NATT
      INTEGER   ETAD,PHID,ETA1,PHI1,ETA2,PHI2,NRE,NRP
      REAL      N, AN1N2, AN1AN2, AN, PMU, C12, ANBIN,FTEMP
      REAL      PI, ETAHI, ETALO, PHIHI, PHILO, DETA, DPHI
      REAL      E_AMPT, PX_AMPT, PY_AMPT, PZ_AMPT, M_AMPT

      parameter (NETA=41)
      parameter (NPHI=41)
      DIMENSION  N(NETA,NPHI),AN(NETA,NPHI)
      DIMENSION  AN1N2(NETA,NPHI),AN1AN2(NETA,NPHI)
      DIMENSION  PMU(130000,4)
      parameter (PI=3.1415926)
      parameter (ETAHI=5.0)
      parameter (ETALO=-5.0)
      parameter (PHIHI=2.0*PI)
      parameter (PHILO=0.0)
      parameter (DETA=(ETAHI-ETALO)/(NETA-1.0))
      parameter (DPHI=(PHIHI-PHILO)/(NPHI-1.0))
C............................................................      
997   format (2I6,F7.4,4I3)
998    FORMAT(1X, 3F10.6, I6, 2F10.6)
      open(100,file='ana/zpc.dat')
C     open(101,file='C12_AuAu_noptcut.dat')
      open(101,file='C12_pion.dat')

      N_EVENT=10
      ETA_CUT=5.0

      ANBIN = 0.0     !<N_bin>
      DO 110 I=1,NETA
      DO 120 J=1,NPHI
          AN(I,J)=0.0
       AN1N2(I,J)=0.0
      AN1AN2(I,J)=0.0
120   CONTINUE
110   CONTINUE


C      WRITE(100,*) N_EVENT
      DO 200 IE=1,N_EVENT

        DO 2110 I=1,NETA
        DO 2120 J=1,NPHI
        N(I,J)=0.0
2120    CONTINUE
2110    CONTINUE

      READ(100,*)ITEMP,NATT,FTEMP,ITEMP,ITEMP,ITEMP,ITEMP
C      WRITE(*,*)ITEMP,NATT,FTEMP,ITEMP,ITEMP,ITEMP,ITEMP

         DO 301 I=1,NATT
         READ(100,*)PX_AMPT,PY_AMPT,PZ_AMPT,ITEMP,FTEMP,E_AMPT
C         WRITE(*,*)PX_AMPT,PY_AMPT,PZ_AMPT,ITEMP,FTEMP,E_AMPT


            PTR=SQRT(PX_AMPT**2+PY_AMPT**2)
C            IF((PTR.LT.0.2)) GO TO 301
            PAB=SQRT(PZ_AMPT**2+PTR**2)
            ETANN=MAX(ABS(PAB+PZ_AMPT),0.0000001)
            ETADD=MAX(ABS(PAB-PZ_AMPT),0.0000001)
            ETA=0.5*ALOG(ETANN/ETADD)
C            IF(ABS(ETA).GT.ETA_CUT) GO TO 301
            PHI=ACOS(PX_AMPT/PTR)
            IF(PY_AMPT.LT.0.0)PHI=2.0*PI-PHI
            IETA=FLOOR((eta-etalo)/deta)+1
            IPHI=FLOOR((PHI-PHILO)/DPHI)+1
            IF(IETA.LT.1.or.IETA.GT.NETA) GO TO 301
            IF(IPHI.LT.1.or.IPHI.GT.NPHI) GO TO 301
C            N(IETA,IPHI) = N(IETA,IPHI)+PTR/DETA/DPHI
            
C            N(IETA,IPHI) = N(IETA,IPHI)+1.0/DETA/DPHI
            N(IETA,IPHI) = N(IETA,IPHI)+1.0
            NTOTAL = NTOTAL +1
 301     CONTINUE


         DO 2003 I=1,NETA
         DO 2004 J=1,NPHI
            AN(I,J) = AN(I,J)+N(I,J)/real(N_EVENT)
2004     CONTINUE
2003     CONTINUE 

         DO 2005 ETAD=0,NETA-1
         DO 2006 PHID=0,NPHI-1
            NRE=NETA-ETAD
            NRP=NPHI-1
            DO 2007 ETA1=0,NRE-1
            DO 2008 PHI1=0,NRP-1
            ETA2=ETA1+ETAD
            PHI2=MOD(PHI1+PHID,NPHI-1)
            AN1N2(ETAD+1,PHID+1) =AN1N2(ETAD+1,PHID+1)+ 
     &   0.5*N(ETA1+1,PHI1+1)*N(ETA2+1,PHI2+1)/real(NRE*NRP*N_EVENT)
     &  +0.5*N(ETA1+1,PHI2+1)*N(ETA2+1,PHI1+1)/real(NRE*NRP*N_EVENT)
2008        CONTINUE
2007        CONTINUE
2006        CONTINUE
2005        CONTINUE

 200  continue

         DO 3005 ETAD=0,NETA-1
         DO 3006 PHID=0,NPHI-1
            NRE=NETA-ETAD
            NRP=NPHI-1
            DO 3007 ETA1=0,NRE-1
            DO 3008 PHI1=0,NRP-1
            ETA2=ETA1+ETAD
            PHI2=MOD(PHI1+PHID,NPHI-1)
            AN1AN2(ETAD+1,PHID+1) =AN1AN2(ETAD+1,PHID+1)+ 
     &       0.5*AN(ETA1+1,PHI1+1)*AN(ETA2+1,PHI2+1)/real(NRE*NRP)
     &      +0.5*AN(ETA1+1,PHI2+1)*AN(ETA2+1,PHI1+1)/real(NRE*NRP)
3008        CONTINUE
3007        CONTINUE
3006        CONTINUE
3005        CONTINUE

         DO 3009 ETAD=1,NETA
         DO 3010 PHID=1,NPHI-1
            if(AN1AN2(ETAD,PHID).gt.1.0d-15)then
            C12 = AN1N2(ETAD,PHID)/AN1AN2(ETAD,PHID)-1.0
            else
            C12 = -1.0
            end if

C            WRITE(101,*)ETAD,PHID,C12
            WRITE(101,*)AN1N2(ETAD,PHID),AN1AN2(ETAD,PHID),C12
3010     CONTINUE
3009     CONTINUE

      Close(100)
      Close(101)
      STOP
      END 
