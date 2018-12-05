#include<iostream>
#include<iomanip>
#include<fstream>
#include<cmath>
#include<cassert>
#include<cstdlib>



using namespace std;
typedef const int CI;
typedef const double CD;
CD pi=3.1415926;

int main(int argc, char** argv)
{
    //assert(argc==2);
    //CD PT_CUT=atof(argv[1]);
    CD PT_CUT=1.0;

    CI NEVENT=100;
    CI NETA=41;
    CI NPHI=41;
    CD ETAHI=5.0;
    CD ETALO=-5.0;
    CD PHIHI=2.0*pi;
    CD PHILO=0.0;
    CD DETA=(ETAHI-ETALO)/(NETA-1.0);
    CD DPHI=(PHIHI-PHILO)/(NPHI-1.0);

    double N[NETA][NPHI]={0.0};
    double AN[NETA][NPHI]={0.0};
    double AN1N2[NETA][NPHI]={0.0};
    double AN1AN2[NETA][NPHI]={0.0};

    //char filename[128];
    //sprintf(filename,"C12_pion%2.1f.dat",PT_CUT);

    //ifstream fzpc("ana/zpc.dat"); 
    ifstream fzpc("ana/parton.oscar"); 
    ofstream fc12("C12_pion.dat");
    char buff[256];
    for(int i=0; i<3; i++)fzpc.getline(buff,256);

    for(int IE=0; IE<NEVENT; IE++){
    char fp4_name[52];
    sprintf(fp4_name,"ana/ampt_b0-5_Ini/P%d.dat",IE);
    ofstream fp4(fp4_name);
    fp4.precision(10);

        for(int I=0; I<NETA; I++)
           for(int J=0; J<NPHI; J++){
              N[I][J] = 0.0;
        }

        int ievent, NATT;
        double temp, Px, Py, Pz, E;
        fzpc>>ievent>>NATT>>temp>>temp;
        fp4<<NATT<<endl; //Output 4 momentum as e-by-e initial condition

        double X, Y, Z, T;
        for(int I=0; I<NATT; I++){
            fzpc>>temp>>temp>>Px>>Py>>Pz>>E>>temp>>X>>Y>>Z>>T;
            fp4<<E<<' '<<Px<<' '<<Py<<' '<<Pz<<' '<<T<<' '<<X<<' '<<Y<<' '<<Z<<endl;
            double Pt=sqrt(Px*Px+Py*Py);
            if(Pt<PT_CUT)continue;
            double Pab=sqrt(Pt*Pt+Pz*Pz);
            double etaNN=max(abs(Pab+Pz),1.0E-7);
            double etaDD=max(abs(Pab-Pz),1.0E-7);
            double eta=0.5*log(etaNN/etaDD);
            double phi=acos(Px/Pt);
            if(Py<0.0)phi=2.0*pi-phi;
            int IETA=int((eta-ETALO)/DETA);
            int IPHI=int((phi-PHILO)/DPHI);
            if(IETA<0 || IETA>NETA-1)continue;
            if(IPHI<0 || IPHI>NPHI-1)continue;
            N[IETA][IPHI] += 1.0;
        }
        fp4.close();

        for(int I=0; I<NETA; I++)
            for(int J=0; J<NPHI; J++){
                AN[I][J] += N[I][J]/double(NEVENT);
            }

        for(int ETAD=0; ETAD<NETA; ETAD++)
            for(int PHID=0; PHID<NPHI-1; PHID++){
                int NRE=NETA-ETAD;
                int NRP=NPHI-1;
                for(int ETA1=0; ETA1<NRE; ETA1++)
                    for(int PHI1=0; PHI1<NRP; PHI1++){
                        int ETA2=ETA1+ETAD;
                        int PHI2=(PHI1+PHID)%(NPHI-1);
                        AN1N2[ETAD][PHID] += (0.5*N[ETA1][PHI1]*N[ETA2][PHI2] + 0.5*N[ETA1][PHI2]*N[ETA2][PHI1])/double(NRE*NRP*NEVENT);
                    }
            }
    }//End for IE

        for(int ETAD=0; ETAD<NETA; ETAD++)
            for(int PHID=0; PHID<NPHI-1; PHID++){
                int NRE=NETA-ETAD;
                int NRP=NPHI-1;
                for(int ETA1=0; ETA1<NRE; ETA1++)
                    for(int PHI1=0; PHI1<NRP; PHI1++){
                        int ETA2=ETA1+ETAD;
                        int PHI2=(PHI1+PHID)%(NPHI-1);
                        AN1AN2[ETAD][PHID] += (0.5*AN[ETA1][PHI1]*AN[ETA2][PHI2] + 0.5*AN[ETA1][PHI2]*AN[ETA2][PHI1])/double(NRE*NRP);
                    }
            }

            double C12;
            for(int ETAD=0; ETAD<NETA; ETAD++)
              for(int PHID=0; PHID<NPHI-1; PHID++){
                 if(AN1AN2[ETAD][PHID]>1.0e-15)C12=AN1N2[ETAD][PHID]/AN1AN2[ETAD][PHID]-1.0;
                 else C12=-1.0;
                 fc12<<AN1N2[ETAD][PHID]<<' '<<AN1AN2[ETAD][PHID]<<' '<<C12<<endl;
                 }

                 fzpc.close();
                 fc12.close();
                   
}
