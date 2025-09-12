#!/usr/bin/env python3

# (C) 2008 Norbert Nemec
# This file is part of the CASINO distribution.
# Permission is given to use the script along with the CASINO program and modify
# it for personal use.

import sys
try:
    import numpy as np
except ImportError:
    sys.exit('This program needs the numpy library, which could not be found.')
try:
    import adfread
except ImportError:
    sys.exit('Could not find internal adfread library.')

############

PLOTCUSPS = False
CUSP_ENFORCE = True

############

data = adfread.adfread()

############

General = data["General"]
Geometry = data["Geometry"]
Properties = data["Properties"]
Basis = data["Basis"]
Core = data["Core"]
Symmetry = data["Symmetry"]

############

Nspins, = General["nspin"]
spin_restricted = (Nspins == 1)

Nvalence_electrons = int(General["electrons"][0])

Natoms, = Geometry["nnuc"]
Natomtypes, = Geometry["ntyp"]
Ndummies, = Geometry["nr of dummy atoms"]
Ndummytypes, = Geometry["nr of dummy atomtypes"]

assert Geometry["nr of atoms"] == Natoms + Ndummies
assert Geometry["nr of atomtypes"] == Natomtypes + Ndummytypes

atyp_idx = Geometry["fragment and atomtype index"].reshape(2,Natoms+Ndummies)[1,:] - 1
assert len(atyp_idx) == Natoms + Ndummies
assert all(0 <= atyp_idx[0:Natoms])
assert all(atyp_idx[0:Natoms] < Natomtypes)
assert all(Natomtypes <= atyp_idx[Natoms:Natoms+Ndummies])
assert all(atyp_idx[Natoms:Natoms+Ndummies] < Natomtypes+Ndummytypes)
atyp_idx = atyp_idx[:Natoms]

total_charge_per_atomtype = Geometry["atomtype total charge"]
atomicnumber_per_atomtype = np.array([ int(c) for c in total_charge_per_atomtype ])
assert all(atomicnumber_per_atomtype[Natomtypes:] == 0)

#####################
#####################

Nharmpoly_per_shelltype = np.array([0,1,4,3,5,7])
Ncartpoly_per_shelltype = np.array([0,1,0,3,6,10])

harm2cart_per_shelltype = [
np.eye(1),  # dummy
np.eye(1),
np.eye(1),  # dummy
np.eye(3),

# from stowfdet code:
#   poly(5)=xy
#   poly(6)=yz
#   poly(7)=zx
#   poly(8)=(3*zz-r(2)) == 2zz-xx-yy
#   poly(9)=(xx-yy)
np.array([
    [ 0., 0., 0.,-1., 1., +++ 1], #x**2
    [ 1., 0., 0., 0., 0., +++ 0], #x*y
    [ 0., 0., 1., 0., 0., +++ 0], #x*z
    [ 0., 0., 0.,-1.,-1., +++ 1], #y**2
    [ 0., 1., 0., 0., 0., +++ 0], #y*z
    [ 0., 0., 0., 2., 0., +++ 1], #z**2
]),

# from stowfdet code:
#    xx_yy3=xx-3*yy
#    xx3_yy=3*xx-yy
#    zz5=5*zz
#    zz5_rr = zz5-r(2)

#    poly(10)=(zz5-3*r(2))*z       ! (2*zz-3*(xx+yy))*z
#    poly(11)=zz5_rr*x             ! (4*zz-(xx+yy))*x
#    poly(12)=zz5_rr*y             ! (4*zz-(xx+yy))*y
#    poly(13)=(xx-yy)*z
#    poly(14)=xy*z
#    poly(15)=xx_yy3*x
#    poly(16)=xx3_yy*y
np.array([
    [ 0.,-1., 0., 0., 0., 1., 0., +++ 1,0,0], #x*x*x
    [ 0., 0.,-1., 0., 0., 0., 3., +++ 0,1,0], #x*x*y
    [-3., 0., 0., 1., 0., 0., 0., +++ 0,0,1], #x*x*z
    [ 0.,-1., 0., 0., 0.,-3., 0., +++ 1,0,0], #x*y*y
    [ 0., 0., 0., 0., 1., 0., 0., +++ 0,0,0], #x*y*z
    [ 0., 4., 0., 0., 0., 0., 0., +++ 1,0,0], #x*z*z
    [ 0., 0.,-1., 0., 0., 0.,-1., +++ 0,1,0], #y*y*y
    [-3., 0., 0.,-1., 0., 0., 0., +++ 0,0,1], #y*y*z
    [ 0., 0., 4., 0., 0., 0., 0., +++ 0,1,0], #y*z*z
    [ 2., 0., 0., 0., 0., 0., 0., +++ 0,0,1], #z*z*z
]),
]

cart2harm_per_shelltype = [ np.linalg.inv(m) for m in harm2cart_per_shelltype ]

#####################
# Valence basis set #
#####################

nbset, = Basis["nbset"]
nbos, = Basis["nbos"]
nbaspt = Basis["nbaspt"]-1
assert nbaspt[0] == 0
assert nbaspt[-1] == nbset

Nvalence_shells_per_atomtype = nbaspt[1:] - nbaspt[:-1]
assert all(Nvalence_shells_per_atomtype >= 0)

Nvalence_shells_per_centre = Nvalence_shells_per_atomtype[atyp_idx]

############

nqbas = Basis["nqbas"]
lqbas = Basis["lqbas"]
alfbas = Basis["alfbas"]
assert len(nqbas) == nbset
assert len(lqbas) == nbset
assert len(alfbas) == nbset

valence_shelltype = lqbas + 1 + (lqbas>0)
valence_shelltype_per_atomtype = [ valence_shelltype[nbaspt[a]:nbaspt[a+1]]
                                   for a in range(Natomtypes) ]

valence_shelltype_per_centre = [ valence_shelltype_per_atomtype[at]
                                 for at in atyp_idx ]

Nvalence_harmbasfns_per_atomtype = [ Nharmpoly_per_shelltype[st].sum()
                                     for st in valence_shelltype_per_atomtype]
Nvalence_harmbasfns_per_centre = [ Nvalence_harmbasfns_per_atomtype[at] for at in atyp_idx ]

valence_order_r = nqbas - lqbas - 1
valence_order_r_per_atomtype = [ valence_order_r[nbaspt[a]:nbaspt[a+1]] for a in range(Natomtypes) ]

valence_zeta = alfbas
valence_zeta_per_atomtype = [ valence_zeta[nbaspt[a]:nbaspt[a+1]] for a in range(Natomtypes) ]

#############

nbptr = Basis["nbptr"]-1
assert nbptr[0] == 0
assert nbptr[-1] == nbos

Nvalence_cartbasfn_per_atomtype = nbptr[1:] - nbptr[:-1]
assert all(Nvalence_cartbasfn_per_atomtype >= 0)

Nvalence_cartbasfn_per_centre = Nvalence_cartbasfn_per_atomtype[atyp_idx]
assert sum(Nvalence_cartbasfn_per_centre) == Basis["naos"]

#############

bnorm = Basis["bnorm"]
assert len(bnorm) == nbos

valence_cartnorm = bnorm
valence_cartnorm_per_atomtype = [ valence_cartnorm[nbptr[a]:nbptr[a+1]] for a in range(Natomtypes) ]

#################
# Core basis set
#################

ncset, = Core["ncset"]
ncorpt = Core["ncorpt"]-1
assert ncorpt[0] == 0
assert ncorpt[-1] == ncset

Ncore_shells_per_atomtype = ncorpt[1:] - ncorpt[:-1]
assert all(Ncore_shells_per_atomtype >= 0)

Ncore_shells_per_centre = Ncore_shells_per_atomtype[atyp_idx]

nrcset = Core["nrcset"].reshape(Natomtypes,4)
assert all(Ncore_shells_per_atomtype == nrcset.sum(axis=1))
assert ncset == nrcset.sum()

############

nqcor = Core["nqcor"]
lqcor = Core["lqcor"]
alfcor = Core["alfcor"]
cornrm = Core["cornrm"]

core_shelltype = lqcor + 1 + (lqcor>0)
core_shelltype_per_atomtype = [ core_shelltype[ncorpt[a]:ncorpt[a+1]] for a in range(Natomtypes) ]

core_order_r = nqcor - lqcor - 1
core_order_r_per_atomtype = [ core_order_r[ncorpt[a]:ncorpt[a+1]] for a in range(Natomtypes) ]

core_zeta = alfcor
core_zeta_per_atomtype = [ core_zeta[ncorpt[a]:ncorpt[a+1]] for a in range(Natomtypes) ]

core_cartnorm = cornrm
core_cartnorm_per_atomtype_per_shell = [ core_cartnorm[ncorpt[a]:ncorpt[a+1]]
                                         for a in range(Natomtypes) ]
core_cartnorm_per_atomtype = []

for at in range(Natomtypes):
    cn = []
    for s in range(Ncore_shells_per_atomtype[at]):
        if core_shelltype_per_atomtype[at][s] == 1:
            cn += [ np.array([core_cartnorm_per_atomtype_per_shell[at][s]]) ]
        elif core_shelltype_per_atomtype[at][s] == 3:
            cn += [ np.array([core_cartnorm_per_atomtype_per_shell[at][s]] * 3) ]
        elif core_shelltype_per_atomtype[at][s] == 4:
            raise ValueError("D type fixed core orbitals not yet implemented")
        elif core_shelltype_per_atomtype[at][s] == 5:
            raise ValueError("F type fixed core orbitals not yet implemented")
        else:
            raise ValueError("unknown shell type")
    if len(cn) > 0:
        core_cartnorm_per_atomtype += [ np.concatenate(cn) ]
    else:
        core_cartnorm_per_atomtype += [ np.zeros([0]) ]

############
############

Nshells_per_centre = Nvalence_shells_per_centre + Ncore_shells_per_centre

shelltype_per_centre = [ np.concatenate([core_shelltype_per_atomtype[at],
                                      valence_shelltype_per_atomtype[at]])
                         for at in atyp_idx ]
order_r_per_centre = [ np.concatenate([core_order_r_per_atomtype[at],
                                    valence_order_r_per_atomtype[at]])
                       for at in atyp_idx ]
zeta_per_centre = [ np.concatenate([core_zeta_per_atomtype[at],
                                 valence_zeta_per_atomtype[at]])
                    for at in atyp_idx ]

for c in range(Natoms):
    assert len(shelltype_per_centre[c]) == Nshells_per_centre[c]
    assert len(order_r_per_centre[c]) == Nshells_per_centre[c]
    assert len(zeta_per_centre[c]) == Nshells_per_centre[c]

###############
# coefficients
###############

Nharmbasfns_per_centre = [ Nharmpoly_per_shelltype[st].sum() for st in shelltype_per_centre]
Nharmbasfns = sum(Nharmbasfns_per_centre)

assert all(Nvalence_cartbasfn_per_centre
           == np.array([ Ncartpoly_per_shelltype[st].sum()
                      for st in valence_shelltype_per_centre ]))
Nvalence_cartbasfn = sum(Nvalence_cartbasfn_per_centre)

####################
# valence orbitals #
####################

nsym, = Symmetry['nsym']
symlab = Symmetry['symlab']
assert len(symlab) == nsym
norb = Symmetry['norb']
assert len(norb) == nsym

def select_coeff(sp):
    """Select valence molecular orbital coefficients for a given spin channel.

    Args:
       sp (int): Spin index (0 for alpha, 1 for beta)

    Returns:
       valence_molorb_cart_coeff (np.ndarray): Array of shape (n_orbs, n_basis),
           containing the selected valence orbital coefficients in Cartesian basis.
   """
    X = ['A','B'][sp]  # Label for spin channel: "A" or "B"

    # Lists to store valence orbital data
    valence_molorb_cart_coeff = []
    valence_molorb_occupation = []
    valence_molorb_eigenvalue = []
    # Dictionary to store leftover partial occupations
    partial_occupations = {}
    # Loop over all symmetries
    for sym in range(nsym):
        Section = data[symlab[sym]]
        # Number of orbitals for this spin and symmetry
        nmo_X, = Section["nmo_" + X]
        assert nmo_X == norb[sym]
        # Fractional occupations for each orbital
        froc_X = Section["froc_" + X]
        assert len(froc_X) == norb[sym]
        # Skip if all occupations are zero
        if all(froc_X == 0.0):
            continue
        # Indices of basis functions for this symmetry
        npart = Section["npart"] - 1
        # Extract molecular orbital coefficients and eigenvalues
        Eigen_Bas_X = Section["Eigen-Bas_"+X].reshape([nmo_X,len(npart)])
        eps_X = Section["eps_"+X].reshape([nmo_X])
        # Loop over all orbitals
        for o in range(nmo_X):
            eigv = eps_X[o]

            valence_molorb_eigenvalue += [eigv]
            occ = froc_X[o]
            # Add any leftover partial occupation for this eigenvalue
            if eigv in partial_occupations:
                occ += partial_occupations.pop(eigv)
            # Check if orbital is considered "occupied"
            if occ + 1e-8 >= 2.0/Nspins:
                valence_molorb_occupation += [ 1 ]
                occ -= 2.0/Nspins
                # Construct coefficient vector in Cartesian basis
                coeff = np.zeros((Nvalence_cartbasfn,))
                coeff[npart] = Eigen_Bas_X[o,:]
                valence_molorb_cart_coeff += [coeff]
            else:
                valence_molorb_occupation += [ 0 ]
            # Store leftover fractional occupation
            if occ>1e-8:
                partial_occupations[eigv] = occ
    # Print any leftover partial occupations
    for k,v in iter(partial_occupations.items()):
        print("spin=",sp,": leftover partial occupation at E=",k,": ",v)
    # Sanity check: should be no leftover occupation
    assert sum(len(p) for p in partial_occupations) == 0

    # Nmolorbs_total = len(valence_molorb_eigenvalue)
    # Number of occupied valence orbitals
    Nmolorbs_occup = len(valence_molorb_cart_coeff)
    assert sum(valence_molorb_occupation) == Nmolorbs_occup
    # Convert lists to NumPy arrays
    valence_molorb_occupation = np.array(valence_molorb_occupation)
    valence_molorb_eigenvalue = np.array(valence_molorb_eigenvalue)
    valence_molorb_cart_coeff = np.array(valence_molorb_cart_coeff)

    # Ensure 2D shape even if only one orbital exists (e.g., hydrogen)
    if valence_molorb_cart_coeff.ndim == 1:
        valence_molorb_cart_coeff = valence_molorb_cart_coeff.reshape(1, -1)

    # Identify occupied and unoccupied orbitals
    occupied = (valence_molorb_occupation[:] == 1)
    occidx = valence_molorb_eigenvalue[occupied]
    unoccidx = valence_molorb_eigenvalue[~occupied]
    # Check HOMO-LUMO ordering (warning if HOMO > LUMO)
    if len(occidx) > 0 and len(unoccidx) > 0:
        HOMO = max(occidx)
        LUMO = min(unoccidx)
        if HOMO > LUMO:
            print("Warning: HOMO > LUMO (may happen in some cases)")
    # Keep only occupied eigenvalues
    valence_molorb_eigenvalue = valence_molorb_eigenvalue[occupied]
    # Sanity check: number of orbitals matches number of coefficients
    assert len(valence_molorb_eigenvalue) == Nmolorbs_occup
    # Sort orbitals by eigenvalue
    order = valence_molorb_eigenvalue.argsort()
    valence_molorb_cart_coeff = valence_molorb_cart_coeff[order, :]

    return valence_molorb_cart_coeff


valence_molorb_cart_coeff = [ select_coeff(sp) for sp in range(Nspins) ]
Nvalence_molorbs = np.array([ c.shape[0] for c in valence_molorb_cart_coeff ])

assert sum(Nvalence_molorbs)*(3-Nspins) == Nvalence_electrons

##############################

cart2harm_matrix = np.zeros((Nharmbasfns,Nvalence_cartbasfn))
cart2harm_constraint = []
i,j = 0,0
for c in range(Natoms):
    at = atyp_idx[c]
    for st in core_shelltype_per_atomtype[at]:
        i+=Nharmpoly_per_shelltype[st]

    for st in valence_shelltype_per_atomtype[at]:
        if st == 1: # S shell
            cart2harm_matrix[i,j] = 1.0
            i+=1;j+=1
        elif st == 3: # P shell
            cart2harm_matrix[i:i+3,j:j+3] = np.eye(3)
            i+=3;j+=3
        elif st == 4: # D shell
            cart2harm_matrix[i:i+5,j:j+6] = cart2harm_per_shelltype[st][:5,:]
            constraint = np.zeros([1,Nvalence_cartbasfn])
            constraint[:,j:j+6] = cart2harm_per_shelltype[st][5:,:]
            cart2harm_constraint += [constraint]
            i+=5;j+=6

        elif st == 5: # F shell
            cart2harm_matrix[i:i+7,j:j+10] = cart2harm_per_shelltype[st][:7,:]
            constraint = np.zeros([3,Nvalence_cartbasfn])
            constraint[:,j:j+10] = cart2harm_per_shelltype[st][7:,:]
            cart2harm_constraint += [constraint]
            i+=7;j+=10

assert i == Nharmbasfns
assert j == Nvalence_cartbasfn

if len(cart2harm_constraint) > 0:
    cart2harm_constraint = np.concatenate(cart2harm_constraint,axis=0)

valence_molorb_harm_coeff = [
    np.zeros((Nharmbasfns,Nvalence_molorbs[sp]))
    for sp in range(Nspins)
]

for sp in range(Nspins):
    for m in range(Nvalence_molorbs[sp]):
        valence_molorb_harm_coeff[sp][:,m] = cart2harm_matrix @ valence_molorb_cart_coeff[sp][m,:]
        if len(cart2harm_constraint) > 0:
            violation = cart2harm_constraint @ valence_molorb_cart_coeff[sp][m,:]
            absviolation = np.sqrt(sum(np.abs(violation**2)))
            if absviolation > 1e-5:
                print("WARNING: cartesian to harmonic conversion: spin #%i, orb #%i "
                      "violated by %g"%(sp,m,absviolation))


#######################
# fixed core orbitals #
#######################

nrcorb = Core["nrcorb"].reshape(Natomtypes,4)
ccor = Core["ccor"]
Nccor_per_atomtype = (nrcset*nrcorb).sum(axis=1)
assert len(ccor) == Nccor_per_atomtype.sum()
ccor_per_atomtype = np.array_split(ccor,np.cumsum(Nccor_per_atomtype))[:-1]

Ncoremolorbs_per_atomtype = (nrcorb*np.array([1,3,5,7])[None,:]).sum(axis=1)
Ncoremolorbs_per_centre = Ncoremolorbs_per_atomtype[atyp_idx]


Ncore_molorbs = Ncoremolorbs_per_centre.sum()

core_molorb_coeff = np.zeros((Nharmbasfns,Ncore_molorbs))

molorb = 0

for a in range(Natoms):
    at = atyp_idx[a]
    first_harmbasfn = sum(Nharmbasfns_per_centre[:a])
    Ncore_harmbasfns = sum(Nharmpoly_per_shelltype[st].sum()
                           for st in core_shelltype_per_atomtype[at])
    core_coeff = np.zeros([Ncore_harmbasfns])
    ccor_per_shell = np.array_split(ccor_per_atomtype[at],np.cumsum((nrcset*nrcorb)[at,:]))[:-1]

    for shell in range(nrcorb[at,0]): # S core shells
        core_coeff[:] = 0.0
        core_coeff[0:nrcset[at,0]] = ccor_per_shell[0][nrcset[at,0]*shell:nrcset[at,0]*(shell+1)]
        core_molorb_coeff[first_harmbasfn:first_harmbasfn+Ncore_harmbasfns,molorb] = core_coeff
        molorb += 1

    for shell in range(nrcorb[at,1]): # P core shells
        for i in range(3):
            core_coeff[:] = 0.0
            offset = nrcset[at,0]
            core_coeff[offset+i:offset+nrcset[at,1]*3:3]\
              = ccor_per_shell[1][nrcset[at,1]*shell:nrcset[at,1]*(shell+1)]
            core_molorb_coeff[first_harmbasfn:first_harmbasfn+Ncore_harmbasfns,molorb] = core_coeff
            molorb += 1
    for shell in range(nrcorb[at,2]): # D core shells
        for i in range(5):
            core_coeff[:] = 0.0
            offset = nrcset[at,0]+nrcset[at,1]
            core_coeff[offset+i:offset+nrcset[at,2]*5:5]\
              = ccor_per_shell[2][nrcset[at,2]*shell:nrcset[at,2]*(shell+1)]
            core_molorb_coeff[first_harmbasfn:first_harmbasfn+Ncore_harmbasfns,molorb] = core_coeff
            molorb += 1
    for shell in range(nrcorb[at,3]): # F core shells
        for i in range(7):
            core_coeff[:] = 0.0
            offset = nrcset[at,0]+nrcset[at,1]+nrcset[at,2]
            core_coeff[offset+i:offset+nrcset[at,2]*7:7]\
              = ccor_per_shell[3][nrcset[at,3]*shell:nrcset[at,3]*(shell+1)]
            core_molorb_coeff[first_harmbasfn:first_harmbasfn+Ncore_harmbasfns,molorb] = core_coeff
            molorb += 1

assert molorb == Ncore_molorbs

############
############

Nmolorbs = np.array([ Ncore_molorbs + Nvalence_molorbs[sp] for sp in range(Nspins) ])
coeff = [
    np.concatenate([core_molorb_coeff,valence_molorb_harm_coeff[sp]],axis=1)
    for sp in range(Nspins)
]

if False:
    print('molorb sparsity:')
    for sp in range(Nspins):
        for i in range(Nmolorbs[sp]):
            print("".join(np.array([".","X"])[(coeff[sp][:,i] != 0.0)*1]))


############
############

norm_per_centre = [ np.concatenate([core_cartnorm_per_atomtype[at],
                                 valence_cartnorm_per_atomtype[at]])
                    for at in atyp_idx ]

norm_per_harmbasfn = np.concatenate(norm_per_centre)
# print(norm_per_centre)
# print(Nharmbasfns)
# assert len(norm_per_harmbasfn) == Nharmbasfns

############
############

try:
    import stowfn
except ImportError:
    sys.exit('Could not find internal stowfn library.')

sto = stowfn.stowfn()

sto.num_atom = Natoms

sto.title, = General['title']
sto.code = 'ADF'
sto.periodicity = 0
sto.spin_unrestricted = not spin_restricted
sto.nuclear_repulsion_energy = 0.0
sto.atomcharge = total_charge_per_atomtype[atyp_idx]
assert len(sto.atomcharge) == Natoms

eionion = 0.0
if Natoms > 1:
    adist = Geometry['Atomic Distances'].reshape(Natoms+1,Natoms+1)[1:,1:]
    for i in range(Natoms):
        assert adist[i,i] == 0.0
        for j in range(i):
            assert adist[i,j] == adist[j,i]
            assert adist[i,j] > 0.0
            eionion += sto.atomcharge[i]*sto.atomcharge[j]/adist[i,j]
sto.nuclear_repulsion_energy = eionion/Natoms

sto.num_elec = Nvalence_electrons + 2*Ncore_molorbs
sto.atompos = Geometry['xyz'].reshape(Natoms+Ndummies,3)[:Natoms,:]
sto.atomnum = atomicnumber_per_atomtype[atyp_idx]

###############
# basis set

sto.num_centres = int(Natoms)
sto.centrepos = Geometry['xyz'].reshape(Natoms+Ndummies,3)[:Natoms,:]
sto.num_shells = sum(Nshells_per_centre)

sto.idx_first_shell_on_centre = np.array([0] + list(np.cumsum(Nshells_per_centre)))

sto.shelltype = np.concatenate(shelltype_per_centre)
sto.order_r_in_shell = np.concatenate(order_r_per_centre)
sto.zeta = np.concatenate(zeta_per_centre)

sto.num_atorbs = Nharmbasfns
sto.num_molorbs = Nmolorbs
sto.coeff = [ c.T for c in coeff ]
sto.footer = ''

sto.check_and_normalize()

# check norm

if False:
    print('Norm-ADF (cartesian)')
    print(norm_per_harmbasfn)
    print('Norm-computed (minimal basis)')
    print(sto.get_norm())
# print(norm_per_harmbasfn - sto.get_norm())

np.set_printoptions(suppress=True)

# assert all(np.abs(norm_per_harmbasfn - sto.get_norm()) < 1e-13)

cusp_fixed_atorbs = sto.cusp_fixed_atorbs()
cusp_constraint = sto.cusp_constraint_matrix()
# print("cusp_constraint_matrix:")
# print(cusp_constraint)
# cusp_projection = sto.cusp_projection_matrix()
cusp_enforcing = sto.cusp_enforcing_matrix()

print('Molorb values at nuclei before applying cusp constraint:')
print(sto.eval_molorbs(sto.atompos.transpose()))

# Initialize a list of boolean masks, one per spin channel.
# Each mask has length equal to the number of molecular orbitals (Nmolorbs[sp]) for that spin.
# Initially, all values are False (no violations detected yet).
fixed = [np.zeros(Nmolorbs[sp], bool) for sp in range(Nspins)]

for sp in range(Nspins):
    for i in range(Nmolorbs[sp]):
        # Compute the cusp constraint violation for orbital i of spin sp
        constraint_violation = cusp_constraint @ coeff[sp][:, i]
        # If any component of the violation is larger than the tolerance (1e-9),
        # we mark this orbital as "fixed" (problematic) and handle it.
        if np.any(np.abs(constraint_violation) > 1e-9):
            fixed[sp][i] = True
            print("spin #%i, orb #%i - constraint violation by:" % (sp, i), constraint_violation)
            if CUSP_ENFORCE:
                # Show original coefficients for the constrained atomic orbitals
                print("    original coefficients:    ",coeff[sp][cusp_fixed_atorbs,i])
                # Projected coefficients (alternative approach, commented out)
                # projected_coeff = cusp_projection.A @ coeff[:,i]
                # print("    proj coeff:",projected_coeff)
                # print("    after projection       :", cusp_constraint @ projected_coeff)
                # Apply the cusp enforcing projection to fix the coefficients
                enforced_coeff = cusp_enforcing.A @ coeff[sp][:,i]
                print("    constrained coefficients: ",enforced_coeff[cusp_fixed_atorbs])
                # Replace the original coefficients with the enforced (corrected) ones
                coeff[sp][:,i] = enforced_coeff
                # Re-check that constraint violation is now within the stricter tolerance
                constraint_violation = cusp_constraint @ coeff[sp][:,i]
                assert np.all(np.abs(constraint_violation) < 1e-8)
                # print("    after enforcing        :", cusp_constraint @ enforced_coeff)

if PLOTCUSPS:
    # Build a z-axis line through each atom from -0.5 to 0.5 (relative units)
    z = np.linspace(-0.5, 0.5, 501)
    r = [np.zeros((3,len(z))) + sto.atompos[at,:][:,None] for at in range(sto.num_atom)]
    for ir in r:
        ir[2, :] += z
    # Print the boolean masks (which orbitals are marked as violating cusp conditions)
    print(fixed)
    val_pre = [[sto.eval_molorbs(ir, spin=sp)[:, fixed[sp]] for sp in range(Nspins)] for ir in r]
    lap_pre = [[sto.eval_molorb_derivs(ir, spin=sp)[2][:, fixed[sp]] for sp in range(Nspins)] for ir in r]

sto.coeff = [ c.T for c in coeff ]
sto.check_and_normalize()

if PLOTCUSPS:
    val_post = [[sto.eval_molorbs(ir, spin=sp)[:, fixed[sp]] for sp in range(Nspins)] for ir in r]
    lap_post = [[sto.eval_molorb_derivs(ir, spin=sp)[2][:, fixed[sp]] for sp in range(Nspins)] for ir in r]

if CUSP_ENFORCE:
    print('Molorb values at nuclei after applying cusp constraint:')
    print(sto.eval_molorbs(sto.atompos.transpose()))
    # assert all(np.abs(norm_per_harmbasfn - sto.get_norm()) < 1e-13)

sto.writefile('stowfn.data')

if PLOTCUSPS:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('The PLOTCUSPS feature requires the matplotlib library, which could not be found.')
        sys.exit()
    Natom = sto.num_atom
    # Create a 2 x Natom grid of subplots
    # Top row: wavefunction values (val)
    # Bottom row: local energies (eloc)
    fig, axes = plt.subplots(2, Natom, figsize=(4*Natom, 4))
    # If only one atom, reshape axes into 2D array for consistency
    if Natom == 1:
        axes = np.array([axes]).reshape(2, 1)
    axval  = [axes[0, at] for at in range(Natom)]
    axeloc = [axes[1, at] for at in range(Natom)]
    for at in range(Natom):
        eloc_min = 1e8
        eloc_max = -1e8
        for sp in range(Nspins):
            for i in range(sum(fixed[sp])):
                vpre = val_pre[at][sp][:, i]  # wavefunction before correction
                vpost = val_post[at][sp][:, i]  # wavefunction after correction
                sgn = np.sign(vpre[len(vpre)//2])  # sign normalization
                # Plot wavefunction before and after correction
                pl, = axval[at].plot(z, sgn*vpre, '--')
                axval[at].plot(z,sgn*vpost,color=pl.get_color())
                # Plot Laplacian of the wavefunction + Coulomb term (without normalization).
                # pl, = axeloc[at].plot(z, lap_pre[at][sp][:,i] + sto.atomnum[at]/z, '--')
                # Compute local energy before and after correction:
                # E_loc = -0.5 * (Laplacian / wavefunction) - Z / |r|
                eloc_pre = -0.5 * lap_pre[at][sp][:,i] / val_pre[at][sp][:, i] - sto.atomnum[at] / np.abs(z)
                eloc_post = -0.5 * lap_post[at][sp][:,i] / val_post[at][sp][:, i] - sto.atomnum[at] / np.abs(z)
                # Plot local energy before and after correction
                axeloc[at].plot(z,eloc_pre,'--',color=pl.get_color())
                axeloc[at].plot(z,eloc_post,'-',color=pl.get_color())
                # Track min/max values for axis scaling
                eloc_min = min(
                    eloc_min,
                    eloc_post[0],
                    eloc_post[-1],
                    eloc_post[len(eloc_post)//2-1],
                    eloc_post[len(eloc_post)//2+1],
                )
                eloc_max = max(
                    eloc_min,
                    eloc_post[0],
                    eloc_post[-1],
                    eloc_post[len(eloc_post)//2-1],
                    eloc_post[len(eloc_post)//2+1],
                )
        # Expand y-limits around the middle value for better visualization
        eloc_mid = (eloc_min + eloc_max) / 2
        eloc_min = (eloc_min - eloc_mid)*1.5 + eloc_mid
        eloc_max = (eloc_max - eloc_mid)*1.5 + eloc_mid
        # Set axis ranges
        axval[at].set_xlim(z[0], z[-1])
        axeloc[at].set_xlim(z[0], z[-1])
        axeloc[at].set_ylim(eloc_min, eloc_max)
    # Adjust layout and save figure
    fig.tight_layout()
    fig.savefig('cusp_constraint.svg')
