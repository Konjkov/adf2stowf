#!/usr/bin/env python3.9
# (C) 2008 Norbert Nemec
# This file is part of the CASINO distribution.
# Permission is given to use the script along with the CASINO program and modify
# it for personal use.

import sys

import numpy as np

from adf2stowf import adfread, cli_main, stowfn

np.set_printoptions(
    suppress=True,
    precision=8,
    floatmode='fixed',
    sign=' ',
)


class ADFToStoWF:
    def __init__(self):
        self.PLOT_CUSPS, self.CUSP_METHOD, self.DO_DUMP, self.CART2HARM_PROJECTION, self.ONLY_OCCUPIED = cli_main.main()
        self.parser = adfread.AdfParser('TAPE21.asc')
        self.data = self.parser.parse()
        self.initialize_data()

    def initialize_data(self):
        # Main data sections from the parsed ADF TAPE21 file
        if self.DO_DUMP:
            self.parser.write_dump('TAPE21.txt')

        self.General = self.data['General']
        self.Geometry = self.data['Geometry']
        self.Properties = self.data['Properties']
        self.Basis = self.data['Basis']
        self.Core = self.data['Core']
        self.Symmetry = self.data['Symmetry']

        self.Nspins = self.General['nspin'][0]
        self.spin_restricted = self.Nspins == 1
        self.Nvalence_electrons = int(self.General['electrons'][0])
        self.Natoms = self.Geometry['nnuc'][0]
        self.Natomtypes = self.Geometry['ntyp'][0]
        self.Ndummies = self.Geometry['nr of dummy atoms'][0]
        self.Ndummytypes = self.Geometry['nr of dummy atomtypes'][0]

        assert self.Geometry['nr of atoms'] == self.Natoms + self.Ndummies
        assert self.Geometry['nr of atomtypes'] == self.Natomtypes + self.Ndummytypes

        self.atyp_idx = self.Geometry['fragment and atomtype index'].reshape(2, self.Natoms + self.Ndummies)[1, :] - 1
        assert len(self.atyp_idx) == self.Natoms + self.Ndummies
        assert np.all(0 <= self.atyp_idx[0 : self.Natoms])
        assert np.all(self.atyp_idx[0 : self.Natoms] < self.Natomtypes)
        assert np.all(self.Natomtypes <= self.atyp_idx[self.Natoms : self.Natoms + self.Ndummies])
        assert np.all(self.atyp_idx[self.Natoms : self.Natoms + self.Ndummies] < self.Natomtypes + self.Ndummytypes)
        self.atyp_idx = self.atyp_idx[: self.Natoms]

        self.total_charge_per_atomtype = self.Geometry['atomtype total charge']
        self.atomicnumber_per_atomtype = np.array([int(c) for c in self.total_charge_per_atomtype])
        assert np.all(self.atomicnumber_per_atomtype[self.Natomtypes :] == 0)

        self.Nharmpoly_per_shelltype = np.array([0, 1, 4, 3, 5, 7])
        self.Ncartpoly_per_shelltype = np.array([0, 1, 0, 3, 6, 10])
        self.harm2cart_map = {
            1: np.eye(1),
            3: np.eye(3),
            4: np.array([
                [0, 0, 0, -1,  1, 1],
                [1, 0, 0,  0,  0, 0],
                [0, 0, 1,  0,  0, 0],
                [0, 0, 0, -1, -1, 1],
                [0, 1, 0,  0,  0, 0],
                [0, 0, 0,  2,  0, 1],
            ]),
            5: np.array([
                [ 0, -1,  0,  0, 0,  1,  0, 1, 0, 0],
                [ 0,  0, -1,  0, 0,  0,  3, 0, 1, 0],
                [-3,  0,  0,  1, 0,  0,  0, 0, 0, 1],
                [ 0, -1,  0,  0, 0, -3,  0, 1, 0, 0],
                [ 0,  0,  0,  0, 1,  0,  0, 0, 0, 0],
                [ 0,  4,  0,  0, 0,  0,  0, 1, 0, 0],
                [ 0,  0, -1,  0, 0,  0, -1, 0, 1, 0],
                [-3,  0,  0, -1, 0,  0,  0, 0, 0, 1],
                [ 0,  0,  4,  0, 0,  0,  0, 0, 1, 0],
                [ 2,  0,  0,  0, 0,  0,  0, 0, 0, 1],
            ])
        }  # fmt: skip
        self.cart2harm_map = {st: np.linalg.inv(M) for st, M in self.harm2cart_map.items()}

    def process_valence_basis(self):
        self.nbset = self.Basis['nbset'][0]
        self.nbos = self.Basis['nbos'][0]
        self.nbaspt = self.Basis['nbaspt'] - 1
        assert self.nbaspt[0] == 0
        assert self.nbaspt[-1] == self.nbset
        self.Nvalence_shells_per_atomtype = self.nbaspt[1:] - self.nbaspt[:-1]
        assert np.all(self.Nvalence_shells_per_atomtype >= 0)
        self.Nvalence_shells_per_centre = self.Nvalence_shells_per_atomtype[self.atyp_idx]

        self.nqbas = self.Basis['nqbas']
        self.lqbas = self.Basis['lqbas']
        self.alfbas = self.Basis['alfbas']
        assert len(self.nqbas) == self.nbset
        assert len(self.lqbas) == self.nbset
        assert len(self.alfbas) == self.nbset
        self.valence_shelltype = self.lqbas + 1 + (self.lqbas > 0)
        self.valence_shelltype_per_atomtype = [self.valence_shelltype[self.nbaspt[a] : self.nbaspt[a + 1]] for a in range(self.Natomtypes)]
        self.valence_shelltype_per_centre = [self.valence_shelltype_per_atomtype[at] for at in self.atyp_idx]
        self.Nvalence_harmbasfns_per_atomtype = [self.Nharmpoly_per_shelltype[st].sum() for st in self.valence_shelltype_per_atomtype]
        self.Nvalence_harmbasfns_per_centre = [self.Nvalence_harmbasfns_per_atomtype[at] for at in self.atyp_idx]
        self.valence_order_r = self.nqbas - self.lqbas - 1
        self.valence_order_r_per_atomtype = [self.valence_order_r[self.nbaspt[a] : self.nbaspt[a + 1]] for a in range(self.Natomtypes)]
        self.valence_zeta = self.alfbas
        self.valence_zeta_per_atomtype = [self.valence_zeta[self.nbaspt[a] : self.nbaspt[a + 1]] for a in range(self.Natomtypes)]

        self.nbptr = self.Basis['nbptr'] - 1
        assert self.nbptr[0] == 0
        assert self.nbptr[-1] == self.nbos
        self.Nvalence_cartbasfn_per_atomtype = self.nbptr[1:] - self.nbptr[:-1]
        assert np.all(self.Nvalence_cartbasfn_per_atomtype >= 0)
        self.Nvalence_cartbasfn_per_centre = self.Nvalence_cartbasfn_per_atomtype[self.atyp_idx]
        assert np.sum(self.Nvalence_cartbasfn_per_centre) == self.Basis['naos']

        self.bnorm = self.Basis['bnorm']
        assert len(self.bnorm) == self.nbos
        self.valence_cartnorm = self.bnorm
        self.valence_cartnorm_per_atomtype = [self.valence_cartnorm[self.nbptr[a] : self.nbptr[a + 1]] for a in range(self.Natomtypes)]

    def process_core_basis(self):
        self.ncset = self.Core['ncset'][0]
        self.ncorpt = self.Core['ncorpt'] - 1
        assert self.ncorpt[0] == 0
        assert self.ncorpt[-1] == self.ncset
        self.Ncore_shells_per_atomtype = self.ncorpt[1:] - self.ncorpt[:-1]
        assert np.all(self.Ncore_shells_per_atomtype >= 0)
        self.Ncore_shells_per_centre = self.Ncore_shells_per_atomtype[self.atyp_idx]
        self.nrcset = self.Core['nrcset'].reshape(self.Natomtypes, 4)
        assert np.all(self.Ncore_shells_per_atomtype == self.nrcset.sum(axis=1))
        assert self.ncset == self.nrcset.sum()

        self.nqcor = self.Core['nqcor']
        self.lqcor = self.Core['lqcor']
        self.alfcor = self.Core['alfcor']
        self.cornrm = self.Core['cornrm']
        self.core_shelltype = self.lqcor + 1 + (self.lqcor > 0)
        self.core_shelltype_per_atomtype = [self.core_shelltype[self.ncorpt[a] : self.ncorpt[a + 1]] for a in range(self.Natomtypes)]
        self.core_order_r = self.nqcor - self.lqcor - 1
        self.core_order_r_per_atomtype = [self.core_order_r[self.ncorpt[a] : self.ncorpt[a + 1]] for a in range(self.Natomtypes)]
        self.core_zeta = self.alfcor
        self.core_zeta_per_atomtype = [self.core_zeta[self.ncorpt[a] : self.ncorpt[a + 1]] for a in range(self.Natomtypes)]
        self.core_cartnorm = self.cornrm
        self.core_cartnorm_per_atomtype_per_shell = [self.core_cartnorm[self.ncorpt[a] : self.ncorpt[a + 1]] for a in range(self.Natomtypes)]
        self.core_cartnorm_per_atomtype = []
        for at in range(self.Natomtypes):
            cn = []
            for s in range(self.Ncore_shells_per_atomtype[at]):
                if self.core_shelltype_per_atomtype[at][s] == 1:
                    cn += [np.array([self.core_cartnorm_per_atomtype_per_shell[at][s]])]
                elif self.core_shelltype_per_atomtype[at][s] == 3:
                    cn += [np.array([self.core_cartnorm_per_atomtype_per_shell[at][s]] * 3)]
                elif self.core_shelltype_per_atomtype[at][s] == 4:
                    raise ValueError('D type fixed core orbitals not yet implemented')
                elif self.core_shelltype_per_atomtype[at][s] == 5:
                    raise ValueError('F type fixed core orbitals not yet implemented')
                else:
                    raise ValueError('unknown shell type')
            if len(cn) > 0:
                self.core_cartnorm_per_atomtype += [np.concatenate(cn)]
            else:
                self.core_cartnorm_per_atomtype += [np.zeros([0])]

    def process_shells(self):
        self.Nshells_per_centre = self.Nvalence_shells_per_centre + self.Ncore_shells_per_centre
        self.shelltype_per_centre = [
            np.concatenate([self.core_shelltype_per_atomtype[at], self.valence_shelltype_per_atomtype[at]]) for at in self.atyp_idx
        ]
        self.order_r_per_centre = [
            np.concatenate([self.core_order_r_per_atomtype[at], self.valence_order_r_per_atomtype[at]]) for at in self.atyp_idx
        ]
        self.zeta_per_centre = [np.concatenate([self.core_zeta_per_atomtype[at], self.valence_zeta_per_atomtype[at]]) for at in self.atyp_idx]
        for c in range(self.Natoms):
            assert len(self.shelltype_per_centre[c]) == self.Nshells_per_centre[c]
            assert len(self.order_r_per_centre[c]) == self.Nshells_per_centre[c]
            assert len(self.zeta_per_centre[c]) == self.Nshells_per_centre[c]

    def select_coeff(self, sp):
        X = ['A', 'B'][sp]
        molorb_cart_coeff = []
        molorb_occupation = []
        molorb_eigenvalue = []
        partial_occupations = {}
        self.nsym = self.Symmetry['nsym'][0]
        self.symlab = self.Symmetry['symlab']
        self.norb = self.Symmetry['norb']
        assert len(self.symlab) == self.nsym
        assert len(self.norb) == self.nsym

        for sym in range(self.nsym):
            Section = self.data[self.symlab[sym]]
            nmo_X = Section['nmo_' + X][0]
            assert nmo_X == self.norb[sym]
            froc_X = Section['froc_' + X]
            assert len(froc_X) == self.norb[sym]
            if np.all(froc_X == 0.0) and self.ONLY_OCCUPIED:
                continue
            npart = Section['npart'] - 1
            Eigen_Bas_X = Section['Eigen-Bas_' + X].reshape([nmo_X, len(npart)])
            eps_X = Section['eps_' + X].reshape([nmo_X])
            for o in range(nmo_X):
                eigv = eps_X[o]
                occ = froc_X[o]
                molorb_eigenvalue.append(eigv)
                if eigv in partial_occupations:
                    occ += partial_occupations.pop(eigv)
                coeff = np.zeros(shape=(self.Nvalence_cartbasfn,))
                coeff[npart] = Eigen_Bas_X[o]
                if occ + 1e-8 >= 2.0 / self.Nspins:
                    molorb_occupation.append(1)
                    occ -= 2.0 / self.Nspins
                    molorb_cart_coeff.append(coeff)
                else:
                    molorb_occupation.append(0)
                    if not self.ONLY_OCCUPIED:
                        molorb_cart_coeff.append(coeff)
                if occ > 1e-8:
                    partial_occupations[eigv] = occ

        for k, v in partial_occupations.items():
            print('spin=', sp, ': leftover partial occupation at E=', k, ': ', v)
        assert len(partial_occupations) == 0

        molorb_occupation = np.array(molorb_occupation)
        molorb_eigenvalue = np.array(molorb_eigenvalue)
        molorb_cart_coeff = np.array(molorb_cart_coeff)
        if molorb_cart_coeff.ndim == 1:
            molorb_cart_coeff = molorb_cart_coeff.reshape(1, -1)

        occupied = molorb_occupation[:] == 1
        occidx = molorb_eigenvalue[occupied]
        unoccidx = molorb_eigenvalue[~occupied]
        if len(occidx) > 0 and len(unoccidx) > 0:
            HOMO = max(occidx)
            LUMO = min(unoccidx)
            if HOMO > LUMO:
                print('Warning: HOMO > LUMO (may happen in some cases)')

        if self.ONLY_OCCUPIED:
            molorb_eigenvalue = molorb_eigenvalue[occupied]
            self.Nmolorbs_occup = len(molorb_cart_coeff)
            assert len(molorb_eigenvalue) == self.Nmolorbs_occup
            assert np.sum(molorb_occupation) == self.Nmolorbs_occup
        else:
            self.Nmolorbs_total = len(molorb_eigenvalue)
            assert molorb_cart_coeff.shape[0] == self.Nmolorbs_total

        order = molorb_eigenvalue.argsort()
        return molorb_cart_coeff[order]

    def process_coefficients(self):
        self.Nharmbasfns_per_centre = [self.Nharmpoly_per_shelltype[st].sum() for st in self.shelltype_per_centre]
        self.Nharmbasfns = np.sum(self.Nharmbasfns_per_centre)
        self.Nvalence_cartbasfn = np.sum(self.Nvalence_cartbasfn_per_centre)
        assert np.all(
            self.Nvalence_cartbasfn_per_centre == np.array([self.Ncartpoly_per_shelltype[st].sum() for st in self.valence_shelltype_per_centre])
        )

        self.molorb_cart_coeff = [self.select_coeff(sp) for sp in range(self.Nspins)]
        self.Nvalence_molorbs = np.array([c.shape[0] for c in self.molorb_cart_coeff])
        if self.ONLY_OCCUPIED:
            assert np.sum(self.Nvalence_molorbs) * (3 - self.Nspins) == self.Nvalence_electrons

        self.cart2harm_matrix = np.zeros((self.Nharmbasfns, self.Nvalence_cartbasfn))
        self.cart2harm_constraint = np.zeros((self.Nvalence_cartbasfn - self.Nharmbasfns, self.Nvalence_cartbasfn))
        i, j = 0, 0
        for atom in range(self.Natoms):
            at = self.atyp_idx[atom]
            for st in self.core_shelltype_per_atomtype[at]:
                i += self.Nharmpoly_per_shelltype[st]
            for shell in range(self.Nvalence_shells_per_atomtype[at]):
                st = self.valence_shelltype_per_atomtype[at][shell]
                n_harm = self.Nharmpoly_per_shelltype[st]
                n_cart = self.Ncartpoly_per_shelltype[st]
                self.cart2harm_matrix[i : i + n_harm, j : j + n_cart] = self.cart2harm_map[st][:n_harm]
                if n_cart > n_harm:
                    constraint = self.cart2harm_map[st][n_harm:]
                    self.cart2harm_constraint[j - i : j - i + n_cart - n_harm, j : j + n_cart] = constraint
                    violation = sum(np.linalg.norm(constraint @ self.molorb_cart_coeff[spin][:, j : j + n_cart].T) for spin in range(self.Nspins))
                    if violation > 1e-5:
                        print(f'WARNING: cartesian to spherical conversion for atom {atom}, shell type {st} violated by {violation:.8f}')
                        current_order_r = self.valence_order_r_per_atomtype[at][shell] + 2
                        current_zeta = self.valence_zeta_per_atomtype[at][shell]
                        print(f'with r_order {current_order_r} and zeta: {current_zeta}')
                i += n_harm
                j += n_cart
        assert i == self.Nharmbasfns
        assert j == self.Nvalence_cartbasfn

        for sp in range(self.Nspins):
            violation = self.cart2harm_constraint @ self.molorb_cart_coeff[sp].T
            absviolations = np.linalg.norm(violation, axis=0)
            for m, err in enumerate(absviolations):
                if err > 1e-5:
                    print(f'WARNING: cartesian to spherical conversion for spin {sp}, orb {m:2d} violated by {err:.8f}')

        if self.CART2HARM_PROJECTION:
            from scipy.linalg import null_space

            A = self.cart2harm_constraint
            Q = null_space(A)
            P = Q @ Q.T
            for sp in range(self.Nspins):
                for m in range(self.Nvalence_molorbs[sp]):
                    C = self.molorb_cart_coeff[sp][m, :].copy()
                    C_proj = P @ C
                    self.molorb_cart_coeff[sp][m, :] = C_proj
                    violation = A @ C_proj
                    absviolation = np.linalg.norm(violation)
                    if absviolation > 1e-10:
                        print(f'WARNING: Projection failed for spin {sp}, orb {m:2d}: {absviolation:.13f}')

        self.valence_molorb_harm_coeff = [self.cart2harm_matrix @ self.molorb_cart_coeff[sp].T for sp in range(self.Nspins)]

    def process_core_orbitals(self):
        self.nrcorb = self.Core['nrcorb'].reshape(self.Natomtypes, 4)
        self.ccor = self.Core['ccor']
        self.Nccor_per_atomtype = (self.nrcset * self.nrcorb).sum(axis=1)
        assert len(self.ccor) == self.Nccor_per_atomtype.sum()
        self.ccor_per_atomtype = np.array_split(self.ccor, np.cumsum(self.Nccor_per_atomtype))[:-1]
        self.Ncoremolorbs_per_atomtype = (self.nrcorb * np.array([1, 3, 5, 7])[None, :]).sum(axis=1)
        self.Ncoremolorbs_per_centre = self.Ncoremolorbs_per_atomtype[self.atyp_idx]
        self.Ncore_molorbs = self.Ncoremolorbs_per_centre.sum()
        self.core_molorb_coeff = np.zeros((self.Nharmbasfns, self.Ncore_molorbs))
        molorb = 0
        for a in range(self.Natoms):
            at = self.atyp_idx[a]
            first_harmbasfn = np.sum(self.Nharmbasfns_per_centre[:a])
            Ncore_harmbasfns = np.sum(self.Nharmpoly_per_shelltype[st].sum() for st in self.core_shelltype_per_atomtype[at])
            core_coeff = np.zeros([Ncore_harmbasfns])
            ccor_per_shell = np.array_split(self.ccor_per_atomtype[at], np.cumsum((self.nrcset * self.nrcorb)[at, :]))[:-1]
            for shell in range(self.nrcorb[at, 0]):
                core_coeff[:] = 0.0
                core_coeff[0 : self.nrcset[at, 0]] = ccor_per_shell[0][self.nrcset[at, 0] * shell : self.nrcset[at, 0] * (shell + 1)]
                self.core_molorb_coeff[first_harmbasfn : first_harmbasfn + Ncore_harmbasfns, molorb] = core_coeff
                molorb += 1
            for shell in range(self.nrcorb[at, 1]):
                for i in range(3):
                    core_coeff[:] = 0.0
                    offset = self.nrcset[at, 0]
                    core_coeff[offset + i : offset + self.nrcset[at, 1] * 3 : 3] = ccor_per_shell[1][
                        self.nrcset[at, 1] * shell : self.nrcset[at, 1] * (shell + 1)
                    ]
                    self.core_molorb_coeff[first_harmbasfn : first_harmbasfn + Ncore_harmbasfns, molorb] = core_coeff
                    molorb += 1
            for shell in range(self.nrcorb[at, 2]):
                for i in range(5):
                    core_coeff[:] = 0.0
                    offset = self.nrcset[at, 0] + self.nrcset[at, 1]
                    core_coeff[offset + i : offset + self.nrcset[at, 2] * 5 : 5] = ccor_per_shell[2][
                        self.nrcset[at, 2] * shell : self.nrcset[at, 2] * (shell + 1)
                    ]
                    self.core_molorb_coeff[first_harmbasfn : first_harmbasfn + Ncore_harmbasfns, molorb] = core_coeff
                    molorb += 1
            for shell in range(self.nrcorb[at, 3]):
                for i in range(7):
                    core_coeff[:] = 0.0
                    offset = self.nrcset[at, 0] + self.nrcset[at, 1] + self.nrcset[at, 2]
                    core_coeff[offset + i : offset + self.nrcset[at, 2] * 7 : 7] = ccor_per_shell[3][
                        self.nrcset[at, 3] * shell : self.nrcset[at, 3] * (shell + 1)
                    ]
                    self.core_molorb_coeff[first_harmbasfn : first_harmbasfn + Ncore_harmbasfns, molorb] = core_coeff
                    molorb += 1
        assert molorb == self.Ncore_molorbs

    def finalize_coefficients(self):
        self.Nmolorbs = np.array([self.Ncore_molorbs + self.Nvalence_molorbs[sp] for sp in range(self.Nspins)])
        self.coeff = [np.concatenate([self.core_molorb_coeff, self.valence_molorb_harm_coeff[sp]], axis=1) for sp in range(self.Nspins)]
        if False:
            print('molorb sparsity:')
            for sp in range(self.Nspins):
                for i in range(self.Nmolorbs[sp]):
                    print(''.join(np.array(['.', 'X'])[(self.coeff[sp][:, i] != 0.0) * 1]))

        self.norm_per_centre = [np.concatenate([self.core_cartnorm_per_atomtype[at], self.valence_cartnorm_per_atomtype[at]]) for at in self.atyp_idx]
        self.norm_per_harmbasfn = np.concatenate(self.norm_per_centre)

    def setup_stowfn(self):
        self.sto = stowfn.StoWfn()
        self.sto.num_atom = self.Natoms
        (self.sto.title,) = self.General['title']
        self.sto.code = 'ADF'
        self.sto.periodicity = 0
        self.sto.spin_unrestricted = not self.spin_restricted
        self.sto.nuclear_repulsion_energy = 0.0
        self.sto.atomcharge = self.total_charge_per_atomtype[self.atyp_idx]
        assert len(self.sto.atomcharge) == self.Natoms
        eionion = 0.0
        if self.Natoms > 1:
            adist = self.Geometry['Atomic Distances'].reshape(self.Natoms + 1, self.Natoms + 1)[1:, 1:]
            for i in range(self.Natoms):
                assert adist[i, i] == 0.0
                for j in range(i):
                    assert adist[i, j] == adist[j, i]
                    assert adist[i, j] > 0.0
                    eionion += self.sto.atomcharge[i] * self.sto.atomcharge[j] / adist[i, j]
            self.sto.nuclear_repulsion_energy = eionion / self.Natoms
        self.sto.num_elec = self.Nvalence_electrons + 2 * self.Ncore_molorbs
        self.sto.atompos = self.Geometry['xyz'].reshape(self.Natoms + self.Ndummies, 3)[: self.Natoms, :]
        self.sto.atomnum = self.atomicnumber_per_atomtype[self.atyp_idx]
        self.sto.num_centres = int(self.Natoms)
        self.sto.centrepos = self.Geometry['xyz'].reshape(self.Natoms + self.Ndummies, 3)[: self.Natoms, :]
        self.sto.num_shells = np.sum(self.Nshells_per_centre)
        self.sto.idx_first_shell_on_centre = np.array([0] + list(np.cumsum(self.Nshells_per_centre)))
        self.sto.shelltype = np.concatenate(self.shelltype_per_centre)
        self.sto.order_r_in_shell = np.concatenate(self.order_r_per_centre)
        self.sto.zeta = np.concatenate(self.zeta_per_centre)
        self.sto.num_atorbs = self.Nharmbasfns
        self.sto.num_molorbs = self.Nmolorbs
        self.sto.footer = ''
        self.sto.coeff = [c.T for c in self.coeff]
        self.sto.check_and_normalize()

    def apply_cusp_correction(self):
        cusp_fixed_atorbs = self.sto.cusp_fixed_atorbs()
        cusp_constraint = self.sto.cusp_constraint_matrix()
        cusp_projection = self.sto.cusp_projection_matrix()
        cusp_enforcing = self.sto.cusp_enforcing_matrix()
        print('Molorb values at nuclei before applying cusp constraint:')
        print(self.sto.eval_molorbs(self.sto.atompos.T))
        self.fixed = [np.zeros(self.Nmolorbs[sp], bool) for sp in range(self.Nspins)]
        for sp in range(self.Nspins):
            for i in range(self.Nmolorbs[sp]):
                constraint_violation = cusp_constraint @ self.coeff[sp][:, i]
                if np.any(np.abs(constraint_violation) > 1e-9):
                    self.fixed[sp][i] = True
                    print(f'spin {sp}, orb {i}:')
                    print('    constraint violation by: ', constraint_violation)
                    print('    original coefficients:   ', self.coeff[sp][cusp_fixed_atorbs, i])
                    if self.CUSP_METHOD == 'project':
                        projected_coeff = cusp_projection @ self.coeff[sp][:, i]
                        print('    projection coefficients:\n', projected_coeff)
                        print('    after projection:         ', cusp_constraint @ projected_coeff)
                        self.coeff[sp][:, i] = projected_coeff
                    if self.CUSP_METHOD == 'enforce':
                        enforced_coeff = cusp_enforcing @ self.coeff[sp][:, i]
                        print('    constrained coefficients:', enforced_coeff[cusp_fixed_atorbs])
                        print('    after enforcing:         ', cusp_constraint @ enforced_coeff)
                        self.coeff[sp][:, i] = enforced_coeff
                if self.CUSP_METHOD != 'none':
                    constraint_violation = cusp_constraint @ self.coeff[sp][:, i]
                    assert np.all(np.abs(constraint_violation) < 1e-8)

        if self.PLOT_CUSPS:
            self.z = np.linspace(-0.5, 0.5, 501)
            r = [np.zeros((3, len(self.z))) + self.sto.atompos[at, :][:, None] for at in range(self.sto.num_atom)]
            for ir in r:
                ir[2, :] += self.z
            self.val_pre = [[self.sto.eval_molorbs(ir, spin=sp)[:, self.fixed[sp]] for sp in range(self.Nspins)] for ir in r]
            self.lap_pre = [[self.sto.eval_molorb_derivs(ir, spin=sp)[2][:, self.fixed[sp]] for sp in range(self.Nspins)] for ir in r]

        self.sto.coeff = [c.T for c in self.coeff]
        self.sto.check_and_normalize()

        if self.PLOT_CUSPS:
            self.val_post = [[self.sto.eval_molorbs(ir, spin=sp)[:, self.fixed[sp]] for sp in range(self.Nspins)] for ir in r]
            self.lap_post = [[self.sto.eval_molorb_derivs(ir, spin=sp)[2][:, self.fixed[sp]] for sp in range(self.Nspins)] for ir in r]

        if self.CUSP_METHOD != 'none':
            print('Molorb values at nuclei after applying cusp constraint:')
            print(self.sto.eval_molorbs(self.sto.atompos.T))

        self.sto.writefile('stowfn.data')

    def plot_cusps(self):
        if not self.PLOT_CUSPS:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('The PLOT_CUSPS feature requires the matplotlib library, which could not be found.')
            sys.exit()

        fig, axes = plt.subplots(2, self.sto.num_atom, figsize=(4 * self.sto.num_atom, 4))
        if self.sto.num_atom == 1:
            axes = np.array([axes]).reshape(2, 1)
        axval = [axes[0, at] for at in range(self.sto.num_atom)]
        axeloc = [axes[1, at] for at in range(self.sto.num_atom)]

        for at in range(self.sto.num_atom):
            eloc_min = 1e8
            eloc_max = -1e8
            for sp in range(self.Nspins):
                for i in range(np.sum(self.fixed[sp])):
                    vpre = self.val_pre[at][sp][:, i]
                    vpost = self.val_post[at][sp][:, i]
                    sgn = np.sign(vpre[len(vpre) // 2])
                    (pl,) = axval[at].plot(self.z, sgn * vpre, '--')
                    axval[at].plot(self.z, sgn * vpost, '-', color=pl.get_color())
                    eloc_pre = -0.5 * self.lap_pre[at][sp][:, i] / self.val_pre[at][sp][:, i] - self.sto.atomnum[at] / np.abs(self.z)
                    eloc_post = -0.5 * self.lap_post[at][sp][:, i] / self.val_post[at][sp][:, i] - self.sto.atomnum[at] / np.abs(self.z)
                    axeloc[at].plot(self.z, eloc_pre, '--', color=pl.get_color())
                    axeloc[at].plot(self.z, eloc_post, '-', color=pl.get_color())
                    eloc_min = min(eloc_min, eloc_post[0], eloc_post[-1], eloc_post[len(eloc_post) // 2 - 1], eloc_post[len(eloc_post) // 2 + 1])
                    eloc_max = max(eloc_min, eloc_post[0], eloc_post[-1], eloc_post[len(eloc_post) // 2 - 1], eloc_post[len(eloc_post) // 2 + 1])
            eloc_mid = (eloc_min + eloc_max) / 2
            eloc_min = (eloc_min - eloc_mid) * 1.5 + eloc_mid
            eloc_max = (eloc_max - eloc_mid) * 1.5 + eloc_mid
            axval[at].set_xlim(self.z[0], self.z[-1])
            axeloc[at].set_xlim(self.z[0], self.z[-1])
            axeloc[at].set_ylim(eloc_min, eloc_max)
            axval[at].set_title(f'Atom {at+1} (Z={self.sto.atomnum[at]}): Wavefunction')
            axeloc[at].set_title(f'Atom {at+1}: Local Energy')

        fig.tight_layout()
        fig.savefig('cusp_constraint.svg')

    def run(self):
        self.process_valence_basis()
        self.process_core_basis()
        self.process_shells()
        self.process_coefficients()
        self.process_core_orbitals()
        self.finalize_coefficients()
        self.setup_stowfn()
        self.apply_cusp_correction()
        self.plot_cusps()


def main():
    adf_to_stowf = ADFToStoWF()
    adf_to_stowf.run()


if __name__ == '__main__':
    main()
