OBJ=main.o amptsub.o art1f.o \
hijing1.383_ampt.o hipyset1.35.o zpc.o linana.o
F77=gfortran

ampt: $(OBJ)
	$(F77) -o ampt -O2 $(OBJ) $(LIBS)

main.o: main.f
	$(F77)  -c -O2 main.f

amptsub.o: amptsub.f
	$(F77)  -c -O2 amptsub.f

art1f.o: art1f.f
	$(F77)  -c -O2 art1f.f

hijing1.383_ampt.o: hijing1.383_ampt.f
	$(F77)  -c -O2 hijing1.383_ampt.f

hipyset1.35.o: hipyset1.35.f
	$(F77)  -c -O2 hipyset1.35.f

zpc.o: zpc.f
	$(F77)  -c -O2 zpc.f

linana.o: linana.f
	$(F77)  -c -O2 linana.f
