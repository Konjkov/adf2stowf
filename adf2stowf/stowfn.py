#!/usr/bin/env python3.9

# (C) 2008 Norbert Nemec
# This file is part of the CASINO distribution.
# Permission is given to use the script along with the CASINO program and modify
# it for personal use.


from adf2stowf.common import *

num_orbs_per_shelltype=numpy.array([0,1,4,3,5,7,9])


support_code = r"""
#line """+'%i'%(lineno()+1)+r""" "stowfn.py"
const double sto_exp_cutoff = 746.0;
const int num_poly_in_shell_type[] = { 0, 1, 4, 3, 5, 7, 9 };
const int first_poly_in_shell_type[] = { 0, 0, 0, 1, 4, 9, 16 };
const double pi = 3.14159265358979323846;

const int polypow[25] = {
    0,
    1,1,1,
    2,2,2,2,2,
    3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4,4
};

double factorial(int N) {
    if(N<=1) return 1.0;
    else     return N*factorial(N-1);
}

template <typename T>
inline blitz::Array<T,1> vec(const T a1,const T a2,const T a3)
{blitz::Array<T,1> res(3); res=a1,a2,a3; return res; }
"""

eval_code = r"""
#line """+'%i'%(lineno()+1)+r""" "stowfn.py"
blitz::Range all = blitz::Range::all();

blitz::Array<double,1> poly(25), phi(9);
#ifdef CALC_DERIVS
blitz::Array<double,2> dpoly(3,25), dphi(3,9);
blitz::Array<double,1> ddphi(9);
#endif

#ifdef EVAL_MOLORBS
blitz::Range M(0,num_molorbs);

val = 0.0;
#ifdef CALC_DERIVS
grad = 0.0;
lap = 0.0;
#endif
#endif

blitz::Array<double,1> r(blitz::Range(-1,max_order_r));

for(int pt=0;pt<num_points;pt++) {
  int n_shell=0;
  int n_atorb=0;

  for(int centre=0;centre<num_centres;centre++) {
    double x=pos(0,pt)-centrepos(centre,0);
    double y=pos(1,pt)-centrepos(centre,1);
    double z=pos(2,pt)-centrepos(centre,2);
    double xx=x*x;
    double yy=y*y;
    double zz=z*z;

    r(2)=xx+yy+zz;
    r(1)=sqrt(r(2));
    for(int i=3;i<=max_order_r_on_centre(centre);i++)
      r(i)=r(i-1)*r(1);

    if(max_order_r_on_centre(centre)>0) {
      r(0)=1.0;
      r(-1)=1.0/r(1);
    }

    double xnorm=x*r(-1);
    double ynorm=y*r(-1);
    double znorm=z*r(-1);

    poly(0) = 1;
    poly(1) = x;
    poly(2) = y;
    poly(3) = z;

#ifdef CALC_DERIVS
    dpoly(all,0) = vec(0.,0.,0.);
    dpoly(all,1) = vec(1.,0.,0.);
    dpoly(all,2) = vec(0.,1.,0.);
    dpoly(all,3) = vec(0.,0.,1.);
#endif

    if(max_shell_type_on_centre(centre)>=4) {
      // there are d and/or higher shells
      double xy = x*y;
      double yz = y*z;
      double zx = z*x;

      // polynomials for d shells
      poly(4)=xy;
      poly(5)=yz;
      poly(6)=zx;
      poly(7)=3*zz-r(2);
      poly(8)=xx-yy;

#ifdef CALC_DERIVS
      dpoly(all,4) = vec(   y,   x,  0.);
      dpoly(all,5) = vec(   0.,   z,  y);
      dpoly(all,6) = vec(   z,   0.,  x);
      dpoly(all,7) = vec(-2*x,-2*y,4*z);
      dpoly(all,8) = vec( 2*x,-2*y,  0.);
#endif

      if(max_shell_type_on_centre(centre) >= 5) {
        // there are f and/or higher shells
        double t1 = 5*zz-r(2);

        poly( 9)=(2*zz-3*(xx+yy))*z;  // (2*zz-3*(xx+yy))*z
        poly(10)=t1*x;                // (4*zz-(xx+yy))*x
        poly(11)=t1*y;                // (4*zz-(xx+yy))*y
        poly(12)=(xx-yy)*z;
        poly(13)=xy*z;
        poly(14)=(xx-3.0*yy)*x;
        poly(15)=(3.0*xx-yy)*y;

#ifdef CALC_DERIVS
        dpoly(all, 9) = vec(-6*zx,-6*yz,6*zz-3*(xx+yy));
        dpoly(all,10) = vec( 4*zz-yy-3*xx,-2*xy,8*zx);
        dpoly(all,11) = vec(-2*xy,4*zz-xx-3*yy,8*yz);
        dpoly(all,12) = vec( 2*zx,-2*yz,xx-yy);
        dpoly(all,13) = vec( yz,zx,xy);
        dpoly(all,14) = vec( 3*xx-3*yy,-6*xy,0.);
        dpoly(all,15) = vec( 6*xy,3*xx-3*yy,0.);
#endif

        if(max_shell_type_on_centre(centre) >= 6) {
          // there are g shells

          double xx_yy3=xx-3*yy;
          double xx3_yy=3*xx-yy;
          double xx_yy=xx-yy;
          double zz5=5*zz;
          double zz7=7*zz;
          double rr3=3*r(2);
          double zz7_rr=zz7-r(2);
          double zz7_rr3=zz7-rr3;

          poly(16)=zz5*(zz7_rr3) - (zz5 - r(2))*rr3; // 35zzzz-30zzrr+3rrrr
          poly(17)=zx*(zz7_rr3);                     // xz(7zz-3rr)
          poly(18)=yz*(zz7_rr3);                     // yz(7zz-3rr)
          poly(19)=(xx_yy)*(zz7_rr);                 // (xx-yy)(7zz-rr)
          poly(20)=xy*(zz7_rr);                      // xy(7zz-rr)
          poly(21)=zx*(xx_yy3);                      // xz(xx-3yy)
          poly(22)=yz*(xx3_yy);                      // yz(3xx-yy)
          poly(23)=xx*(xx_yy3) - yy*(xx3_yy);        // xxxx-6xxyy+yyyy
          poly(24)=xy*(xx_yy);                       // xxxy-xyyy


#ifdef CALC_DERIVS
//          dpoly(all, 16) = -60zzx+12rrx,   -60zzy+12rry,   140zzz-60zrr-60zzz+12rrz
//          dpoly(all, 17) = 7zzz-3zrr-6zxx, -6xyz,          21xzz-3xrr-6xzz
//          dpoly(all, 18) = -6xyz,          7zzz-3zrr-6zyy, 21yzz-3yrr-6yzz
//          dpoly(all, 19) = 14xzz-2xrr-2xxx+2xyy, ...
//          dpoly(all, 20) =
//          dpoly(all, 21) =
//          dpoly(all, 22) =
//          dpoly(all, 23) =
//          dpoly(all, 24) =

          throw "gradient and laplacian of g orbitals not yet implemented";
#endif
        }
      }
    }

//std::cout << "poly = "<< poly << "\n";
//#ifdef CALC_DERIVS
//std::cout << "dpoly = "<< dpoly << "\n";
//#endif

    for(int shell=0; shell<num_shells_on_centre(centre);shell++,n_shell++) {
      double zeta_rabs=zeta(n_shell)*r(1);
      if(zeta_rabs>sto_exp_cutoff) {
        n_atorb += num_poly_in_shell_type[shelltype(n_shell)];
        continue; // to next shell
      }
      double exp_zeta_rabs=exp(-zeta_rabs);
      blitz::Range A(first_poly_in_shell_type[shelltype(n_shell)],first_poly_in_shell_type[shelltype(n_shell)]+num_poly_in_shell_type[shelltype(n_shell)]-1);
      blitz::Range X(0,num_poly_in_shell_type[shelltype(n_shell)]-1);
      phi(X) = poly(A)*exp_zeta_rabs;
#ifdef CALC_DERIVS
      dphi(0,X) = dpoly(0,A)*exp_zeta_rabs-zeta(n_shell)*xnorm*phi(X);
      dphi(1,X) = dpoly(1,A)*exp_zeta_rabs-zeta(n_shell)*ynorm*phi(X);
      dphi(2,X) = dpoly(2,A)*exp_zeta_rabs-zeta(n_shell)*znorm*phi(X);
      ddphi(X) = (
          -2*zeta(n_shell)*r(-1)*(x*dpoly(0,A)+y*dpoly(1,A)+z*dpoly(2,A)+poly(A))
          +zeta(n_shell)*zeta(n_shell)*poly(A)
        )*exp_zeta_rabs;
#endif

      int N=order_r_in_shell(n_shell);

      if(N==0) {

#ifdef EVAL_ATORBS
        for(int pl=0;pl<num_poly_in_shell_type[shelltype(n_shell)];pl++) {
          atorbs(pt,n_atorb)=phi(pl);
          n_atorb++;
        }
#endif // EVAL_ATORBS

#ifdef EVAL_MOLORBS
        for(int pl=0;pl<num_poly_in_shell_type[shelltype(n_shell)];pl++) {
          val(pt,all)+=coeff_norm(all,n_atorb)*phi(pl);
#ifdef CALC_DERIVS
          grad(0,pt,all)+=coeff_norm(all,n_atorb)*dphi(0,pl);
          grad(1,pt,all)+=coeff_norm(all,n_atorb)*dphi(1,pl);
          grad(2,pt,all)+=coeff_norm(all,n_atorb)*dphi(2,pl);
          lap(pt,all)+=coeff_norm(all,n_atorb)*ddphi(pl);
#endif
          n_atorb++;
        }
#endif // EVAL_MOLORBS

      } else {

#ifdef EVAL_ATORBS
        for(int pl=0;pl<num_poly_in_shell_type[shelltype(n_shell)];pl++) {
          atorbs(pt,n_atorb)=r(N)*phi(pl);
          n_atorb++;
        }
#endif // EVAL_ATORBS

#ifdef EVAL_MOLORBS
        for(int pl=0;pl<num_poly_in_shell_type[shelltype(n_shell)];pl++) {
          val(pt,all)+=coeff_norm(all,n_atorb)*r(N)*phi(pl);
#ifdef CALC_DERIVS
          grad(0,pt,all)+=coeff_norm(all,n_atorb)*(N*x*r(N-2)*phi(pl)+r(N)*dphi(0,pl));
          grad(1,pt,all)+=coeff_norm(all,n_atorb)*(N*y*r(N-2)*phi(pl)+r(N)*dphi(1,pl));
          grad(2,pt,all)+=coeff_norm(all,n_atorb)*(N*z*r(N-2)*phi(pl)+r(N)*dphi(2,pl));
          lap(pt,all)+=coeff_norm(all,n_atorb)*
            (N*(N+1)*r(N-2)*phi(pl)+2*N*r(N-2)*(x*dphi(0,pl)+y*dphi(1,pl)+z*dphi(2,pl))+r(N)*ddphi(pl));
#endif
          n_atorb++;
        }
#endif // EVAL_MOLORBS

      }

    } // shell
  } // centre
} // pos
"""


norm_code = r"""
#line """+'%i'%(lineno()+1)+r""" "stowfn.py"
int n_shell=0;
int n_atorb=0;
double polynorm[25];

polynorm[0] = sqrt(1./(4.*pi)); // 1
polynorm[1] = sqrt(3./(4.*pi)); // x
polynorm[2] = sqrt(3./(4.*pi)); // y
polynorm[3] = sqrt(3./(4.*pi)); // z

polynorm[4] = .5*sqrt(15./pi); // xy
polynorm[5] = .5*sqrt(15./pi); // yz
polynorm[6] = .5*sqrt(15./pi); // zx
polynorm[7] = .25*sqrt(5./pi); // 3*zz-r(2);
polynorm[8] = .25*sqrt(15./pi); // xx-yy;

polynorm[ 9] = .25*sqrt(7./pi); // (2*zz-3*(xx+yy))*z;
polynorm[10] = .25*sqrt(10.5/pi); // (4*zz-(xx+yy))*x;
polynorm[11] = .25*sqrt(10.5/pi); // (4*zz-(xx+yy))*y;
polynorm[12] = .25*sqrt(105./pi); // (xx-yy)*z;
polynorm[13] = .5*sqrt(105./pi); // xy*z;
polynorm[14] = .25*sqrt(17.5/pi); // (xx-3.0*yy)*x;
polynorm[15] = .25*sqrt(17.5/pi); // (3.0*xx-yy)*y;

polynorm[16] = .1875*sqrt(1./pi); // 35zzzz-30zzrr+3rrrr
polynorm[17] = .75*sqrt(2.5/pi); // xz(7zz-3rr)
polynorm[18] = .75*sqrt(2.5/pi); // yz(7zz-3rr)
polynorm[19] = .375*sqrt(5./pi); // (xx-yy)(7zz-rr)
polynorm[20] = .75*sqrt(5./pi); // xy(7zz-rr)
polynorm[21] = .75*sqrt(17.5/pi); // xz(xx-3yy)
polynorm[22] = .75*sqrt(17.5/pi); // yz(3xx-yy)
polynorm[23] = .1875*sqrt(35./pi); // xxxx-6xxyy+yyyy
polynorm[24] = .75*sqrt(35./pi); // xxxy-xyyy

for(int centre=0; centre<num_centres;centre++) {
    for(int shell=0; shell<num_shells_on_centre(centre); shell++,n_shell++) {
        for(int pl=first_poly_in_shell_type[shelltype(n_shell)];
            pl < first_poly_in_shell_type[shelltype(n_shell)]+num_poly_in_shell_type[shelltype(n_shell)];
            pl++, n_atorb++) {
            int n = polypow[pl] + order_r_in_shell(n_shell) + 1;
            norm(n_atorb)=polynorm[pl] * pow(2*zeta(n_shell),n) * sqrt(2*zeta(n_shell)/factorial(2*n));
        } // pl
    } // shell
} // centre
"""


class stowfn:
    def __init__(self,fname=None):
        if fname is not None:
            self.readfile(fname)
        else:
            self.initempty()

    def initempty(self):
        pass

    def readfile(self,fname):
        f=file(fname)

        def readline():
            return f.readline()
        def readstr():
            return readline().strip()
        def readint():
            return int(readstr())
        def readfloat():
            return float(readstr())
        def readfloats(N):
            res = []
            while len(res) < N:
                l = readline()
                res += [ float(l[i:i+20]) for i in range(0,len(l)-1,20) ]
            assert len(res) == N
            return numpy.array(res)
        def readints(N):
            res = []
            while len(res) < N:
                res += [ int(n) for n in readline().split() ]
            assert len(res) == N
            return numpy.array(res)
        def readbool():
            return F2P_bool[readstr()]
        def skipline(text=""):
            l=f.readline()
            assert l==text+"\n"

        self.title=readstr()
        skipline()
        skipline("BASIC INFO")
        skipline("----------")
        skipline("Generated by:")
        self.code = readstr()
        skipline("Periodicity:")
        self.periodicity = readint()
        skipline("Spin unrestricted:")
        self.spin_unrestricted = readbool()
        skipline("Nuclear repulsion energy (au/atom):")
        self.nuclear_repulsion_energy = readfloat()
        skipline("Number of electrons")
        self.num_elec = readint()
        skipline()
        skipline("GEOMETRY")
        skipline("--------")
        skipline("Number of atoms")
        self.num_atom = readint()
        skipline("Atomic positions (au)")
        self.atompos=readfloats(self.num_atom*3).reshape((self.num_atom,3))
        skipline("Atomic numbers for each atom")
        self.atomnum = readints(self.num_atom)
        skipline("Valence charges for each atom")
        self.atomcharge = readfloats(self.num_atom)
        skipline()
        skipline("BASIS SET")
        skipline("---------")
        skipline("Number of STO centres")
        self.num_centres = readint()
        skipline("Position of each centre (au)")
        self.centrepos = readfloats(self.num_centres*3).reshape((self.num_centres,3))
        skipline("Number of shells")
        self.num_shells = readint()
        skipline("Sequence number of first shell on each centre")
        self.idx_first_shell_on_centre = numpy.array(list(readints(self.num_centres) - 1) + [self.num_shells])
        skipline("Code for shell types (s/sp/p/d/f/g 1/2/3/4/5/6)")
        self.shelltype = readints(self.num_shells)
        skipline("Order of radial prefactor r in each shell")
        self.order_r_in_shell = readints(self.num_shells)
        skipline("Exponent in each STO shell")
        self.zeta = readfloats(self.num_shells)
        skipline("Number of basis functions ('AO')")
        self.num_atorbs = readint()
        skipline("Number of molecular orbitals ('MO')")
        self.num_molorbs = readints(1+self.spin_unrestricted)
        skipline()

        assert self.idx_first_shell_on_centre[-1] == self.num_shells
        assert self.idx_first_shell_on_centre[0] == 0

        self.num_shells_on_centre = self.idx_first_shell_on_centre[1:] - self.idx_first_shell_on_centre[:-1]
        self.max_order_r_on_centre = numpy.array([
            max(2,self.order_r_in_shell[self.idx_first_shell_on_centre[i]:self.idx_first_shell_on_centre[i+1]].max())
            for i in range(self.num_centres)
        ])
        self.max_order_r = self.max_order_r_on_centre.max()
        self.max_shell_type_on_centre = numpy.array([
            self.shelltype[self.idx_first_shell_on_centre[i]:self.idx_first_shell_on_centre[i+1]].max()
            for i in range(self.num_centres)
        ])
        assert all(self.num_shells_on_centre > 0)
        assert sum(num_orbs_per_shelltype[self.shelltype]) == self.num_atorbs

        skipline("MULTIDETERMINANT INFORMATION")
        skipline("----------------------------")
        skipline("GS")
        skipline()

        line = readline()
        if line == "ORBITAL COEFFICIENTS (normalized AO)\n":
            skipline("------------------------------------")
            self.coeff = [ readfloats(self.num_molorbs[0]*self.num_atorbs).reshape((self.num_molorbs[0],self.num_atorbs)) ]
            if self.spin_unrestricted:
                self.coeff += [ readfloats(self.num_molorbs[1]*self.num_atorbs).reshape((self.num_molorbs[1],self.num_atorbs)) ]
            self.coeff_norm = [ c[:,:] * self.get_norm()[None,:] for c in self.coeff ]
            skipline()
            self.footer = f.readlines()
        else:
            self.coeff = [ numpy.zeros((self.num_molorbs[0],self.num_atorbs)) ]
            if self.spin_unrestricted:
                self.coeff += [ numpy.zeros((self.num_molorbs[1],self.num_atorbs)) ]
            self.coeff_norm = [ c[:,:]*0.0 for c in self.coeff ]
            self.footer = [line] + f.readlines()
        f.close()

    def check_and_normalize(self):
        self.title            = str(self.title)
        self.code              = str(self.code)
        self.periodicity       = int(self.periodicity)
        self.spin_unrestricted = bool(self.spin_unrestricted)
        self.nuclear_repulsion_energy = float(self.nuclear_repulsion_energy)
        self.num_elec = int(self.num_elec)
        self.num_atom = int(self.num_atom)
        assert self.atompos.shape == (self.num_atom,3)
        assert numpy.issubdtype(self.atompos.dtype,float)
        assert self.atomnum.shape == (self.num_atom,)
        assert numpy.issubdtype(self.atomnum.dtype,int)
        assert self.atomcharge.shape == (self.num_atom,)
        assert numpy.issubdtype(self.atomcharge.dtype,float)
        self.num_centres = int(self.num_centres)
        assert self.centrepos.shape == (self.num_centres,3)
        assert numpy.issubdtype(self.centrepos.dtype,float)
        self.num_shells = int(self.num_shells)
        assert self.idx_first_shell_on_centre.shape == (self.num_centres+1,)
        assert numpy.issubdtype(self.idx_first_shell_on_centre.dtype,int)
        assert self.shelltype.shape == (self.num_shells,)
        assert numpy.issubdtype(self.shelltype.dtype,int)
        assert self.order_r_in_shell.shape == (self.num_shells,)
        assert numpy.issubdtype(self.order_r_in_shell.dtype,int)
        assert self.zeta.shape == (self.num_shells,)
        assert numpy.issubdtype(self.zeta.dtype,float)
        self.num_atorbs = int(self.num_atorbs)
        assert self.num_molorbs.shape == (1+self.spin_unrestricted,)

        assert self.idx_first_shell_on_centre[-1] == self.num_shells
        assert self.idx_first_shell_on_centre[0] == 0

        self.num_shells_on_centre = self.idx_first_shell_on_centre[1:] - self.idx_first_shell_on_centre[:-1]
        self.max_order_r_on_centre = numpy.array([
            max(2,self.order_r_in_shell[self.idx_first_shell_on_centre[i]:self.idx_first_shell_on_centre[i+1]].max())
            for i in range(self.num_centres)
        ])
        self.max_order_r = self.max_order_r_on_centre.max()
        self.max_shell_type_on_centre = numpy.array([
            self.shelltype[self.idx_first_shell_on_centre[i]:self.idx_first_shell_on_centre[i+1]].max()
            for i in range(self.num_centres)
        ])
        assert all(self.num_shells_on_centre > 0)
        assert sum(num_orbs_per_shelltype[self.shelltype]) == self.num_atorbs

        self.num_spins = 1+self.spin_unrestricted
        assert len(self.coeff) == self.num_spins
        for sp in range(self.num_spins):
            assert self.coeff[sp].shape == (self.num_molorbs[sp],self.num_atorbs)
            assert self.coeff[sp].dtype == numpy.float64
        self.coeff_norm = [
            self.coeff[sp][:,:] * self.get_norm()[None,:]
            for sp in range(self.num_spins)
        ]


    def writefile(self,fname):
        f = open(fname,"w")

        def writeline(l=""):
            f.write(l+"\n")
        def writestr(s):
            writeline(" "+s)
        def writeint(i):
            writestr("%i"%i)
        def writefloat(f):
            writestr("%.15f"%f)
        def writefloats(F,num_cols=4):
            N = len(F)
            for c in range(0,N-num_cols+1,num_cols):
                writeline(("% .13E"*num_cols)%tuple(F[c:c+num_cols]))
            if N%num_cols:
                writeline(("% .13E"*(N%num_cols))%tuple(F[-(N%num_cols):]))
        def writeints(I,num_cols=8):
            N = len(I)
            for c in range(0,N-num_cols+1,num_cols):
                writeline(("% 10i"*num_cols)%tuple(I[c:c+num_cols]))
            if N%num_cols:
                writeline(("% 10i"*(N%num_cols))%tuple(I[-(N%num_cols):]))
        def writebool(b):
            return writestr(P2F_bool[b])

        writeline(self.title)
        writeline()
        writeline("BASIC INFO")
        writeline("----------")
        writeline("Generated by:")
        writestr(self.code)
        writeline("Periodicity:")
        writeint(self.periodicity)
        writeline("Spin unrestricted:")
        writebool(self.spin_unrestricted)
        writeline("Nuclear repulsion energy (au/atom):")
        writefloat(self.nuclear_repulsion_energy)
        writeline("Number of electrons")
        writeint(self.num_elec)
        writeline()
        writeline("GEOMETRY")
        writeline("--------")
        writeline("Number of atoms")
        writeint(self.num_atom)
        writeline("Atomic positions (au)")
        writefloats(self.atompos.reshape((self.num_atom*3)),num_cols=3)
        writeline("Atomic numbers for each atom")
        writeints(self.atomnum)
        writeline("Valence charges for each atom")
        writefloats(self.atomcharge)
        writeline()
        writeline("BASIS SET")
        writeline("---------")
        writeline("Number of STO centres")
        writeint(self.num_centres)
        writeline("Position of each centre (au)")
        writefloats(self.centrepos.reshape((self.num_centres*3)),num_cols=3)
        writeline("Number of shells")
        writeint(self.num_shells)
        writeline("Sequence number of first shell on each centre")
        writeints(self.idx_first_shell_on_centre[:-1]+1)
        writeline("Code for shell types (s/sp/p/d/f/g 1/2/3/4/5/6)")
        writeints(self.shelltype)
        writeline("Order of radial prefactor r in each shell")
        writeints(self.order_r_in_shell)
        writeline("Exponent in each STO shell")
        writefloats(self.zeta)
        writeline("Number of basis functions ('AO')")
        writeint(self.num_atorbs)
        writeline("Number of molecular orbitals ('MO')")
        writeints(self.num_molorbs)
        writeline()
        writeline("MULTIDETERMINANT INFORMATION")
        writeline("----------------------------")
        writeline("GS")
        writeline()

        if hasattr(self,"coeff_norm"):
            writeline("ORBITAL COEFFICIENTS (normalized AO)")
            writeline("------------------------------------")
            coeff = self.coeff_norm[0][:,:] / self.get_norm()[None,:]
            writefloats(coeff.reshape((self.num_molorbs[0]*self.num_atorbs)))
            if self.spin_unrestricted:
                coeff = self.coeff_norm[1][:,:] / self.get_norm()[None,:]
                writefloats(coeff.reshape((self.num_molorbs[1]*self.num_atorbs)))
            writeline()
        elif hasattr(self,"coeff"):
            writeline("ORBITAL COEFFICIENTS (normalized AO)")
            writeline("------------------------------------")
            writefloats(self.coeff[0].reshape((self.num_molorbs[0]*self.num_atorbs)))
            if self.spin_unrestricted:
                writefloats(self.coeff[1].reshape((self.num_molorbs[1]*self.num_atorbs)))
            writeline()

        f.writelines(self.footer)
        f.close()

    def read_molorbmods(self,fname="correlation.data"):
        l = [ l.strip() for l in open(fname,"r").readlines() ]
        start = l.index("START MOLORBMODS")+3
        end = l.index("END MOLORBMODS")
        while start < end:
            if l[start] == "START MOLECULAR ORBITAL COEFFICIENTS":
                endcoeff = l.index("END MOLECULAR ORBITAL COEFFICIENTS")
                moc = [ float(c.split()[0]) for c in l[start+2:endcoeff] ]
                start = endcoeff+1
            elif l[start] == "START STO EXPONENT ZETAS":
                endzeta = l.index("END STO EXPONENT ZETAS")
                zet = [ float(c.split()[0]) for c in l[start+2:endzeta] ]
                start = endzeta+1
            else:
                raise "unknown block starting with '"+l[start]+"'"

    def eval_molorbs(self,pos,spin=0):
        num_points = pos.shape[1]
        assert pos.shape == (3,num_points)
        num_molorbs = self.num_molorbs[spin]
        val = numpy.zeros((num_points,num_molorbs))
        coeff_norm = self.coeff_norm[spin]
        dict = mapunion(self.__dict__,locals())
        weave_inline(support_code,eval_code,dict,["EVAL_MOLORBS"])
        return val

    def eval_molorb_derivs(self,pos,spin=0):
        num_points = pos.shape[1]
        assert pos.shape == (3,num_points)
        num_molorbs = self.num_molorbs[spin]
        val = numpy.zeros((num_points,num_molorbs))
        grad = numpy.zeros((3,num_points,num_molorbs))
        lap = numpy.zeros((num_points,num_molorbs))
        coeff_norm = self.coeff_norm[spin]
        dict = mapunion(self.__dict__,locals())
        weave_inline(support_code,eval_code,dict,["EVAL_MOLORBS","CALC_DERIVS"])
        return val,grad,lap

    def eval_atorbs(self,pos):
        num_points = pos.shape[1]
        assert pos.shape == (3,num_points)
        atorbs = numpy.zeros((num_points,self.num_atorbs))
        dict = mapunion(self.__dict__,locals())
        weave_inline(support_code,eval_code,dict,["EVAL_ATORBS"])
        return atorbs

    def get_norm(self):
        norm = numpy.zeros((self.num_atorbs,))
        dict = mapunion(self.__dict__,locals())
        weave_inline(support_code,norm_code,dict)
        return norm

    def iter_atorbs(self):
        nshell = 0
        atorb = 0
        for centre in range(self.num_centres):
            for _shell in range(self.num_shells_on_centre[centre]):
                for pl in range(num_orbs_per_shelltype[self.shelltype[nshell]]):
                    yield (atorb,centre,nshell,self.order_r_in_shell[nshell],pl)
                    atorb += 1
                nshell += 1

    def cusp_constraint_matrix(self):
        norm = self.get_norm()
        res = numpy.asmatrix(numpy.zeros((self.num_centres,self.num_atorbs)))
        for core in range(self.num_centres):
            atorb_vals = self.eval_atorbs(self.centrepos[core][:,None])[0,:]
            for (atorb,centre,nshell,N,_pl) in self.iter_atorbs():
                if centre == core:
                    if self.shelltype[nshell] == 1:
                        if N == 0:
                            res[core,atorb] = norm[atorb] * (self.atomcharge[core] - self.zeta[nshell])
                        elif N == 1:
                            res[core,atorb] = norm[atorb]
                else: # centre != core
                    res[core,atorb] = norm[atorb]*self.atomcharge[core]*atorb_vals[atorb]
        return res

    def cusp_projection_matrix(self):
        #print "cusp_constraint: ",cusp_constraint
        _U,_S,Vh = numpy.linalg.svd(self.cusp_constraint_matrix(),full_matrices=False)
        #print "shapes U,S,Vh",U.shape,S.shape,Vh.shape
        P = Vh.T * Vh
        #print "proj shape",cusp_constraint_projector.shape
        #print "proj trace",P.trace()
        #print "proj squarediff",numpy.linalg.norm(P - P*P)
        Q = numpy.eye(P.shape[0]) - P
        return Q

    def cusp_fixed_atorbs(self):
        res = numpy.zeros(self.num_centres,int)
        for c in range(self.num_centres):
            cidx = numpy.zeros(self.num_shells)
            cidx[self.idx_first_shell_on_centre[c]:self.idx_first_shell_on_centre[c+1]] = 1.0
            shell = numpy.argmax(self.zeta * cidx * (self.shelltype == 1))
            res[c] = num_orbs_per_shelltype[self.shelltype[:shell]].sum()
        return res

    def cusp_enforcing_matrix(self):
        cusp_fixed_atorb = self.cusp_fixed_atorbs()
        constraint = self.cusp_constraint_matrix()
        res = constraint + 0.0
        res[:,cusp_fixed_atorb] = 0.0
        U,S,Vh = numpy.linalg.svd(constraint[:,cusp_fixed_atorb],full_matrices=False)
        tmpinv = Vh.T * numpy.asmatrix(numpy.diag(1/S)) * U.T
        res = -tmpinv * res
        mat = numpy.asmatrix(numpy.eye(self.num_atorbs))
        mat[cusp_fixed_atorb,:] = res
        return mat


if __name__ == "__main__":
    sto = stowfn("stowfn.data")
    sto.read_molorbmods("correlation.data")
    points = numpy.zeros((3,4))
    points[:,0] = (-0.19450689,-0.94412413,-0.67370571)
    points[:,:] = points[:,:1]
    points[0,1] += 0.00317100
    points[1,2] += 0.00317100
    points[2,3] += 0.00317100
    val,grad,lap = sto.eval_molorb_derivs(points)
    print("grad analytic:",grad[:,0])
    print("grad numeric:",(val[1:]-val[0])/0.00317100)
#    print sto.get_norm()
#    sto.writefile("stowfn.data.out")
