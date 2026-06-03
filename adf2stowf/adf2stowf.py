#!/usr/bin/env python3

import argparse

import numpy as np

from adf2stowf import adfread, stowfn

np.set_printoptions(
    suppress=True,
    precision=8,
    floatmode='fixed',
    sign=' ',
)


class ADFToStoWF:
    def __init__(self, plot_cusps, cusp_method, do_dump, cart2harm_projection, only_occupied):
        """Initialize the ADFToStoWF object."""
        self.do_plot_cusps = plot_cusps
        self.cusp_method = cusp_method
        self.do_dump = do_dump
        self.cart2harm_projection = cart2harm_projection
        self.only_occupied = only_occupied
        self.parser = adfread.AdfParser('TAPE21.asc')
        self.data = self.parser.parse()
        self.initialize_data()

    def initialize_data(self):
        """Load TAPE21 sections into named attributes and optionally dump them."""
        if self.do_dump:
            self.parser.write_dump('TAPE21.txt')

        self.General = self.data['General']
        self.Geometry = self.data['Geometry']
        self.Properties = self.data['Properties']
        self.Basis = self.data['Basis']
        self.Core = self.data['Core']
        self.Symmetry = self.data['Symmetry']

        self._parse_system()
        self.Nharmpoly_per_shelltype, self.Ncartpoly_per_shelltype, self.harm2cart_map, self.cart2harm_map = self._build_cart2harm_maps()

    def _parse_system(self):
        """Extract scalar system metadata and atom-type index from Geometry/General."""
        self.Nspins = self.General['nspin'][0]
        self.spin_restricted = self.Nspins == 1
        self.Nvalence_electrons = int(self.General['electrons'][0])
        self.Natoms = self.Geometry['nnuc'][0]
        self.Natomtypes = self.Geometry['ntyp'][0]
        self.Ndummies = self.Geometry['nr of dummy atoms'][0]
        self.Ndummytypes = self.Geometry['nr of dummy atomtypes'][0]

        assert self.Geometry['nr of atoms'] == self.Natoms + self.Ndummies
        assert self.Geometry['nr of atomtypes'] == self.Natomtypes + self.Ndummytypes

        # atyp_idx: zero-based atom-type index for each atom (real + dummy)
        atyp_idx = self.Geometry['fragment and atomtype index'].reshape(2, self.Natoms + self.Ndummies)[1, :] - 1
        assert len(atyp_idx) == self.Natoms + self.Ndummies
        assert np.all(0 <= atyp_idx[: self.Natoms])
        assert np.all(atyp_idx[: self.Natoms] < self.Natomtypes)
        assert np.all(self.Natomtypes <= atyp_idx[self.Natoms :])
        assert np.all(atyp_idx[self.Natoms :] < self.Natomtypes + self.Ndummytypes)
        self.atyp_idx = atyp_idx[: self.Natoms]  # drop dummy atoms

        self.total_charge_per_atomtype = self.Geometry['atomtype total charge']
        self.atomicnumber_per_atomtype = np.array([int(c) for c in self.total_charge_per_atomtype])
        assert np.all(self.atomicnumber_per_atomtype[self.Natomtypes :] == 0)

        self.nsym = self.Symmetry['nsym'][0]
        self.symlab = self.Symmetry['symlab']
        self.norb = self.Symmetry['norb']
        assert len(self.symlab) == self.nsym
        assert len(self.norb) == self.nsym

    @staticmethod
    def _build_cart2harm_maps():
        """Return shell-type lookup tables and Cartesian<->spherical transformation matrices.

        The harm2cart matrices follow the CASINO stowfdet.f90 polynomial ordering:

        D-shell (shelltype=4), Cartesian order: xy, xz, yz, xx-yy, 2zz-xx-yy, S-contam
          D(-2)=xy  D(-1)=yz  D(1)=xz  D(0)=2zz-xx-yy  D(2)=xx-yy  [S=x2+y2+z2]

        F-shell (shelltype=5), Cartesian order: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
          F(0)=2zzz-3(xxz+yyz)  F(1)=4xzz-(xx+yy)x  F(-1)=4yzz-(xx+yy)y
          F(2)=(xx-yy)z  F(-2)=xyz  F(3)=xxx-3xyy  F(-3)=3xxy-yyy
          [P_x, P_y, P_z contaminants]
        """
        # Number of spherical / Cartesian functions per shell type code
        # Index: shelltype code (0=unused, 1=s, 2=sp, 3=p, 4=d, 5=f)
        Nharmpoly = np.array([0, 1, 4, 3, 5, 7])
        Ncartpoly = np.array([0, 1, 0, 3, 6, 10])

        harm2cart = {
            # S-shell: trivial identity
            1: np.eye(1),
            # P-shell: trivial identity
            3: np.eye(3),
            # D-shell: 5 spherical rows + 1 s-contamination row
            4: np.array([
                [0, 0, 0, -1,  1, 1],
                [1, 0, 0,  0,  0, 0],
                [0, 0, 1,  0,  0, 0],
                [0, 0, 0, -1, -1, 1],
                [0, 1, 0,  0,  0, 0],
                [0, 0, 0,  2,  0, 1],
            ]),
            # F-shell: 7 spherical rows + 3 p-contamination rows
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
            ]),
        }  # fmt: skip

        cart2harm = {st: np.linalg.inv(M) for st, M in harm2cart.items()}
        return Nharmpoly, Ncartpoly, harm2cart, cart2harm

    def process_valence_basis(self):
        """Parse the valence STO basis from the TAPE21 Basis section.

        Reads shell counts, angular momenta, principal quantum numbers, zeta
        exponents, and Cartesian normalisation factors for all valence shells.
        Derives per-atom-type and per-centre views of every quantity, as well
        as the shell-type codes and radial prefactor orders (order_r = n - l - 1)
        required by CASINO.

        Sets attributes (among others):
            nbset, nbos                        – total shell / Cartesian-function counts
            nbaspt, nbptr                      – 0-based shell / Cartesian-fn pointers per atom type
            nqbas, lqbas, alfbas               – raw TAPE21 arrays (n, l, ζ)
            valence_shelltype                  – CASINO shelltype code per shell (s=1, p=3, d=4, f=5)
            valence_order_r, valence_zeta      – radial prefactor exponent and ζ per shell
            valence_cartnorm                   – ADF Cartesian normalisation per basis function
            Nvalence_shells_per_atomtype/centre – shell counts for layout bookkeeping
            Nvalence_cartbasfn_per_atomtype/centre – Cartesian function counts
            Nvalence_harmbasfns_per_atomtype/centre – spherical harmonic function counts
        """
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
        """Parse the frozen-core STO basis from the TAPE21 Core section.

        Reads shell counts, angular momenta, principal quantum numbers, zeta
        exponents, and Cartesian normalisation factors for all frozen-core shells.
        Constructs per-atom-type and per-centre views analogous to those built
        by ``process_valence_basis``.

        Only s-type (shelltype=1) and p-type (shelltype=3) core shells are
        supported; d- and f-type core shells raise ``ValueError``.

        Sets attributes (among others):
            ncset                              – total number of core STO shells
            ncorpt                             – 0-based shell pointer per atom type
            ccor                               – raw flattened core MO coefficient array from TAPE21
            nqcor, lqcor, alfcor, cornrm       – raw TAPE21 arrays (n, l, ζ, norm)
            nrcset                             – shape (Natomtypes, 4): shell counts per l
            core_shelltype                     – CASINO shelltype code per core shell
            core_order_r, core_zeta            – radial prefactor exponent and ζ per core shell
            core_cartnorm_per_atomtype         – Cartesian normalisation per atom type (flat)
            Ncore_shells_per_atomtype/centre   – core shell counts for layout bookkeeping
        """
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

        self.ccor = self.Core['ccor']
        self.nqcor = self.Core['nqcor']
        self.lqcor = self.Core['lqcor']
        self.alfcor = self.Core['alfcor']
        self.cornrm = self.Core['cornrm']
        self.core_shelltype = self.lqcor + 1 + (self.lqcor > 0)
        self.core_shelltype_per_atomtype = [self.core_shelltype[self.ncorpt[a] : self.ncorpt[a + 1]] for a in range(self.Natomtypes)]
        self.core_order_r = self.nqcor - self.lqcor - 1
        self.core_order_r_per_atomtype = [self.core_order_r[self.ncorpt[a] : self.ncorpt[a + 1]] for a in range(self.Natomtypes)]
        self.core_zeta_per_atomtype = [self.alfcor[self.ncorpt[a] : self.ncorpt[a + 1]] for a in range(self.Natomtypes)]
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
        """Merge core and valence shell data into per-centre arrays.

        Concatenates the per-atom-type core and valence arrays (shell type,
        radial-prefactor order, and zeta exponent) into per-centre lists that
        cover *all* shells on each atom in the order CASINO expects: core
        shells first, followed by valence shells.

        Requires that both ``process_core_basis`` and ``process_valence_basis``
        have already been called.

        Sets attributes:
            Nshells_per_centre     – total number of shells on each atom
            shelltype_per_centre   – list of CASINO shelltype arrays, one per atom
            order_r_per_centre     – list of radial-prefactor exponent arrays, one per atom
            zeta_per_centre        – list of ζ-exponent arrays, one per atom
        """
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
        """Extract and sort molecular-orbital coefficients for one spin channel.

        Iterates over all ADF symmetry irreps and collects Cartesian-basis MO
        coefficient vectors, orbital eigenvalues, and occupation numbers for
        spin channel ``sp`` (0 = alpha / restricted, 1 = beta).

        Handles fractional occupations (e.g. from Fermi smearing) by carrying
        over residual partial occupation to the next orbital with the same
        eigenvalue.  A warning is printed for any unresolved leftover occupation
        after all irreps have been processed.

        When ``self.only_occupied`` is True, only fully occupied MOs
        (occupation ≥ 2/Nspins) are returned; otherwise all MOs are returned.

        Returns:
            molorb_cart_coeff (ndarray, shape (Nmo, Ncart)):
                Cartesian-basis coefficient matrix sorted by orbital eigenvalue,
                where Nmo is the number of selected MOs and Ncart is the total
                number of Cartesian AO basis functions.
        """
        X = ['A', 'B'][sp]
        molorb_cart_coeff = []
        molorb_occupation = []
        molorb_eigenvalue = []
        partial_occupations = {}
        assert len(self.symlab) == self.nsym
        assert len(self.norb) == self.nsym
        # Loop over all symmetries
        for sym in range(self.nsym):
            Section = self.data[self.symlab[sym]]
            # Number of orbitals for this spin and symmetry
            nmo_X = Section['nmo_' + X][0]
            assert nmo_X == self.norb[sym]
            # Fractional occupations for each orbital
            froc_X = Section['froc_' + X]
            assert len(froc_X) == self.norb[sym]
            # Skip this symmetry only when we want only occupied orbitals
            if np.all(froc_X == 0.0) and self.only_occupied:
                continue
            # Indices of basis functions for this symmetry
            npart = Section['npart'] - 1
            # Extract molecular orbital coefficients and eigenvalue
            Eigen_Bas_X = Section['Eigen-Bas_' + X].reshape([nmo_X, len(npart)])
            eps_X = Section['eps_' + X].reshape([nmo_X])
            # Loop over all orbitals
            for o in range(nmo_X):
                eigv = eps_X[o]
                occ = froc_X[o]
                molorb_eigenvalue.append(eigv)
                # Add any leftover partial occupation for this eigenvalue
                if eigv in partial_occupations:
                    occ += partial_occupations.pop(eigv)
                coeff = np.zeros(shape=(self.Nvalence_cartbasfn,))
                coeff[npart] = Eigen_Bas_X[o]
                # Check if orbital is considered "occupied"
                if occ + 1e-8 >= 2.0 / self.Nspins:
                    molorb_occupation.append(1)
                    occ -= 2.0 / self.Nspins
                    # Construct coefficient vector in Cartesian basis
                    molorb_cart_coeff.append(coeff)
                else:
                    molorb_occupation.append(0)
                    if not self.only_occupied:
                        molorb_cart_coeff.append(coeff)
                # Store leftover fractional occupation
                if occ > 1e-8:
                    partial_occupations[eigv] = occ
        # Print any leftover partial occupations
        for k, v in partial_occupations.items():
            print('spin=', sp, ': leftover partial occupation at E=', k, ': ', v)
        assert len(partial_occupations) == 0
        # Convert lists to NumPy arrays
        molorb_occupation = np.array(molorb_occupation)
        molorb_eigenvalue = np.array(molorb_eigenvalue)
        molorb_cart_coeff = np.array(molorb_cart_coeff)
        # Ensure 2D shape even if only one orbital exists (e.g., hydrogen)
        if molorb_cart_coeff.ndim == 1:
            molorb_cart_coeff = molorb_cart_coeff.reshape(-1, 1)
        # Identify occupied and unoccupied orbitals
        occupied = molorb_occupation[:] == 1
        occidx = molorb_eigenvalue[occupied]
        unoccidx = molorb_eigenvalue[~occupied]
        # Check HOMO-LUMO ordering (warning if HOMO > LUMO)
        if len(occidx) > 0 and len(unoccidx) > 0:
            HOMO = max(occidx)
            LUMO = min(unoccidx)
            if HOMO > LUMO:
                print('Warning: HOMO > LUMO (may happen in some cases)')
        if self.only_occupied:
            # Keep only occupied eigenvalues
            molorb_eigenvalue = molorb_eigenvalue[occupied]
            # Number of occupied valence orbitals
            self.Nmolorbs_occup = len(molorb_cart_coeff)
            # Sanity check: number of orbitals matches number of coefficients
            assert len(molorb_eigenvalue) == self.Nmolorbs_occup
            assert np.sum(molorb_occupation) == self.Nmolorbs_occup
        else:
            self.Nmolorbs_total = len(molorb_eigenvalue)
            # when returning all orbitals, ensure we have coefficients for each eigenvalue appended
            assert molorb_cart_coeff.shape[0] == self.Nmolorbs_total
        order = molorb_eigenvalue.argsort()
        # Sort orbitals by eigenvalue
        return molorb_cart_coeff[order]

    def process_coefficients(self):
        """Transform MO coefficients from the Cartesian AO basis to spherical harmonics.

        Builds the block-diagonal ``cart2harm_matrix`` (shape Nharmbasfns × Ncart)
        and the complementary ``cart2harm_constraint`` matrix (shape
        (Ncart - Nharmbasfns) × Ncart) that captures non-physical Cartesian
        components (s-contamination in d-shells, p-contamination in f-shells).

        A warning is printed for each shell or orbital where the constraint
        violation norm exceeds 1e-5, indicating that the Cartesian MO expansion
        contains physically spurious components that will be lost in the
        projection to spherical harmonics unless
        ``extract_contamination_shells`` is called afterwards.

        Sets attributes:
            Nharmbasfns               – total number of spherical-harmonic basis functions
            Nvalence_cartbasfn        – total number of Cartesian valence basis functions
            molorb_cart_coeff         – list (per spin) of Cartesian MO coefficient matrices
            Nvalence_molorbs          – array of MO counts per spin channel
            cart2harm_matrix          – Cartesian → spherical transformation (block-diagonal)
            cart2harm_constraint      – rows of cart2harm_map beyond n_harm (contamination extractor)
            valence_molorb_harm_coeff – list (per spin) of MO coefficients in spherical-harmonic basis,
                                        shape (Nharmbasfns, Nmo) per spin
        """
        self.Nharmbasfns_per_centre = [self.Nharmpoly_per_shelltype[st].sum() for st in self.shelltype_per_centre]
        self.Nharmbasfns = np.sum(self.Nharmbasfns_per_centre)
        self.Nvalence_cartbasfn = np.sum(self.Nvalence_cartbasfn_per_centre)
        assert np.all(
            self.Nvalence_cartbasfn_per_centre == np.array([self.Ncartpoly_per_shelltype[st].sum() for st in self.valence_shelltype_per_centre])
        )

        self.molorb_cart_coeff = [self.select_coeff(sp) for sp in range(self.Nspins)]
        self.Nvalence_molorbs = np.array([c.shape[0] for c in self.molorb_cart_coeff])
        if self.only_occupied:
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
                    active_spins = [sp for sp in range(self.Nspins) if self.molorb_cart_coeff[sp].shape[0] > 0]
                    if active_spins:
                        violation = sum(np.linalg.norm(constraint @ self.molorb_cart_coeff[sp][:, j : j + n_cart].T) for sp in active_spins)
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
            if self.molorb_cart_coeff[sp].shape[0] == 0:
                continue
            violation = self.cart2harm_constraint @ self.molorb_cart_coeff[sp].T
            absviolations = np.linalg.norm(violation, axis=0)
            for m, err in enumerate(absviolations):
                if err > 1e-5:
                    print(f'WARNING: cartesian to spherical conversion for spin {sp}, orb {m:2d} violated by {err:.8f}')

        if self.cart2harm_projection:
            # Use SVD-based nullspace for numerical stability (preferred over direct pseudoinverse)
            from scipy.linalg import null_space

            # Compute projection matrix: P = I - A^T (A A^T)^{-1} A
            A = self.cart2harm_constraint
            # Compute orthonormal basis for the nullspace of A (i.e., vectors x such that A @ x = 0)
            Q = null_space(A)  # Q: (Ncart, Ncart - K), columns = orthonormal basis of nullspace
            P = Q @ Q.T  # Projection matrix: P @ x projects x onto the nullspace of A
            # Apply projection to each molecular orbital
            for sp in range(self.Nspins):
                for m in range(self.Nvalence_molorbs[sp]):
                    # Original Cartesian orbital coefficient vector
                    C = self.molorb_cart_coeff[sp][m, :].copy()
                    # Project onto pure spherical harmonic subspace
                    C_proj = P @ C
                    # Overwrite with projected coefficients
                    self.molorb_cart_coeff[sp][m, :] = C_proj
                    # Verify constraint satisfaction: should be numerically zero
                    violation = A @ C_proj
                    absviolation = np.linalg.norm(violation)
                    if absviolation > 1e-10:
                        print(f'WARNING: Projection failed for spin {sp}, orb {m:2d}: {absviolation:.13f}')

        self.valence_molorb_harm_coeff = [
            self.cart2harm_matrix @ self.molorb_cart_coeff[sp].T if self.molorb_cart_coeff[sp].shape[0] > 0 else np.zeros((self.Nharmbasfns, 0))
            for sp in range(self.Nspins)
        ]

    def process_core_orbitals(self):
        """Reconstruct frozen-core MO coefficient matrix in the spherical-harmonic basis.

        ADF stores core MO coefficients (``ccor``) per atom type and per angular
        momentum shell, in a compact interleaved format.  This method unpacks
        them into a full ``core_molorb_coeff`` matrix of shape
        (Nharmbasfns, Ncore_molorbs), placing each core orbital's coefficients
        at the rows corresponding to its host atom.

        Core orbitals are produced for each atom of that type by iterating over
        ``atyp_idx``.  For each angular-momentum channel (s, p, d, f) the
        number of MOs is taken from ``nrcorb``, and the coefficients are
        unpacked via vectorised reshape + advanced indexing:

          For l-channel with ``n_harm`` harmonics (1/3/5/7 for s/p/d/f) and
          ``Nshells`` basis shells, the raw ``ccor`` block has shape
          ``(Norbs, Nshells)`` and maps to interleaved rows
          ``offset + i + s * n_harm`` (i = harmonic index, s = shell index)
          for each of the ``Norbs * n_harm`` output MO columns.

        Sets attributes:
            nrcorb                     – shape (Natomtypes, 4): core MO counts per l per atom type
            Ncoremolorbs_per_atomtype  – total number of core MOs per atom type (counting harmonics)
            Ncoremolorbs_per_centre    – core MO count mapped to each atom
            Ncore_molorbs              – grand total of core MOs across all atoms
            core_molorb_coeff          – full coefficient matrix (Nharmbasfns × Ncore_molorbs)
        """
        self.nrcorb = self.Core['nrcorb'].reshape(self.Natomtypes, 4)
        self.Nccor_per_atomtype = (self.nrcset * self.nrcorb).sum(axis=1)
        assert len(self.ccor) == self.Nccor_per_atomtype.sum()
        self.ccor_per_atomtype = np.array_split(self.ccor, np.cumsum(self.Nccor_per_atomtype))[:-1]
        self.Ncoremolorbs_per_atomtype = self.nrcorb @ np.array([1, 3, 5, 7])
        self.Ncoremolorbs_per_centre = self.Ncoremolorbs_per_atomtype[self.atyp_idx]
        self.Ncore_molorbs = self.Ncoremolorbs_per_centre.sum()
        self.core_molorb_coeff = np.zeros((self.Nharmbasfns, self.Ncore_molorbs))

        # n_harm per l-channel: s=1, p=3, d=5, f=7
        n_harm_per_l = np.array([1, 3, 5, 7])
        # Precompute per-atomtype: cumulative harmbasfn offset of each l-block within core region
        # offset_l[at, l] = sum of n_shells[at, 0..l-1] * n_harm[0..l-1]
        core_harmbasfn_offsets = np.zeros((self.Natomtypes, 4), dtype=int)
        for l in range(1, 4):
            core_harmbasfn_offsets[:, l] = core_harmbasfn_offsets[:, l - 1] + self.nrcset[:, l - 1] * n_harm_per_l[l - 1]

        molorb = 0
        first_harmbasfns = np.concatenate([[0], np.cumsum(self.Nharmbasfns_per_centre)])
        for atom in range(self.Natoms):
            at = self.atyp_idx[atom]
            ccor_per_shell = np.array_split(self.ccor_per_atomtype[at], np.cumsum((self.nrcset * self.nrcorb)[at, :]))[:-1]

            for l, n_harm in enumerate(n_harm_per_l):
                Nshells = self.nrcset[at, l]  # basis shells for this l
                Norbs = self.nrcorb[at, l]  # MOs for this l
                if Norbs == 0:
                    continue
                # ccor_per_shell[l]: flat array of shape (Norbs * Nshells,)
                # Reshape to (Norbs, Nshells), then expand harmonics → (Norbs, n_harm, Nshells)
                # The target block in core_molorb_coeff has shape (Ncore_harmbasfns, Norbs * n_harm)
                # and the l-block rows are offset + i with stride n_harm (interleaved layout):
                #   row offset_l + i + n_harm*s  for s in 0..Nshells-1, i in 0..n_harm-1
                coeffs = ccor_per_shell[l].reshape(Norbs, Nshells)  # (Norbs, Nshells)
                offset = core_harmbasfn_offsets[at, l]
                # Build interleaved index: for each harmonic component i, rows are
                # offset + i, offset + n_harm + i, ..., offset + (Nshells-1)*n_harm + i
                shell_rows = offset + np.arange(Nshells) * n_harm  # (Nshells,)
                for i in range(n_harm):
                    # coeffs[:, :]  has shape (Norbs, Nshells)
                    # column molorb + i*Norbs + orb  ← orb-th MO, component i
                    # We write all Norbs MOs for component i at once:
                    self.core_molorb_coeff[first_harmbasfns[atom] + shell_rows + i, molorb + np.arange(Norbs) * n_harm + i] = coeffs.T
                molorb += Norbs * n_harm

        assert molorb == self.Ncore_molorbs

    def finalize_coefficients(self):
        """Assemble the full MO coefficient matrix and normalisation vector.

        Concatenates the frozen-core and valence MO coefficient blocks along
        the orbital (column) axis for each spin channel, producing a single
        ``coeff`` matrix of shape (Nharmbasfns, Ncore_molorbs + Nvalence_molorbs).

        When ``extract_contamination_shells`` has been called, the valence
        block may have extra rows (contamination-shell coefficients appended
        below the spherical-harmonic rows).  Any missing rows are zero-padded
        to Nharmbasfns so the concatenation is always well-formed.

        Also builds ``norm_per_harmbasfn``, the flat array of ADF Cartesian
        normalisation factors (one entry per spherical-harmonic basis function
        across all centres) used later by ``StoWfn.check_and_normalize``.

        Sets attributes:
            Nmolorbs            – array of total MO counts per spin (core + valence)
            coeff               – list (per spin) of full coefficient matrices
            norm_per_centre     – list (per atom) of concatenated core+valence norms
            norm_per_harmbasfn  – flat norm array over all basis functions
        """
        self.Nmolorbs = np.array([self.Ncore_molorbs + self.Nvalence_molorbs[sp] for sp in range(self.Nspins)])
        self.coeff = [np.concatenate([self.core_molorb_coeff, self.valence_molorb_harm_coeff[sp]], axis=1) for sp in range(self.Nspins)]
        self.norm_per_centre = [np.concatenate([self.core_cartnorm_per_atomtype[at], self.valence_cartnorm_per_atomtype[at]]) for at in self.atyp_idx]
        self.norm_per_harmbasfn = np.concatenate(self.norm_per_centre)

    def setup_stowfn(self):
        """Populate a StoWfn object with geometry, basis, and MO data.

        Transfers all data assembled by the previous processing steps into a
        fresh ``stowfn.StoWfn`` instance (``self.sto``), ready for cusp
        correction and final file output.

        Also computes the nuclear repulsion energy from pairwise interatomic
        distances (averaged over atoms as stored in TAPE21) and sets the
        total electron count to include both valence and core electrons.

        After population, calls ``self.sto.check_and_normalize()`` to verify
        consistency and apply any remaining normalisation.

        Sets attribute:
            sto – fully populated StoWfn object
        """
        self.sto = stowfn.StoWfn()
        self.sto.num_atom = self.Natoms
        self.sto.title = self.General['title'][0]
        self.sto.code = 'ADF'
        self.sto.periodicity = 0
        self.sto.spin_unrestricted = not self.spin_restricted
        self.sto.nuclear_repulsion_energy = 0.0
        self.sto.atomcharge = self.total_charge_per_atomtype[self.atyp_idx]
        assert len(self.sto.atomcharge) == self.Natoms
        eionion = 0.0
        print(self.Geometry['Atomic Distances'])
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
        self.sto.atompos = self.sto.centrepos = self.Geometry['xyz'].reshape(self.Natoms + self.Ndummies, 3)[: self.Natoms, :]
        self.sto.atomnum = self.atomicnumber_per_atomtype[self.atyp_idx]
        self.sto.num_centres = self.Natoms
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
        """Apply or check the nuclear cusp condition on each molecular orbital.

        For every MO and each spin channel, evaluates the cusp constraint
        (``sto.cusp_constraint_matrix() @ coeff[:, i]``).  If the constraint
        is violated beyond 1e-9, the orbital is flagged and — depending on
        ``self.cusp_method``:

        * ``'project'``: projects out the cusp-violating component via a
          null-space projection matrix (``sto.cusp_projection_matrix()``).
        * ``'enforce'``: applies a linear enforcing matrix
          (``sto.cusp_enforcing_matrix()``) that modifies the s-type
          coefficients at each nucleus to satisfy the cusp exactly.
        * ``'none'``:  no modification; violation is only recorded.

        After correction, asserts that all constraint violations are < 1e-8.

        When ``self.do_plot_cusps`` is True, evaluates the wavefunction values
        and Laplacians along a z-axis line through each atom both before and
        after correction, storing them in ``val_pre / val_post`` and
        ``lap_pre / lap_post`` for later use by ``plot_cusps()``.

        Finally, transposes ``self.coeff`` into ``sto.coeff`` (shape
        (Nmo, Nharmbasfns)) and calls ``sto.check_and_normalize()`` and
        ``sto.writefile('stowfn.data')``.
        """
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
                    if self.cusp_method == 'project':
                        projected_coeff = cusp_projection @ self.coeff[sp][:, i]
                        print('    projection coefficients:\n', projected_coeff)
                        print('    after projection:         ', cusp_constraint @ projected_coeff)
                        self.coeff[sp][:, i] = projected_coeff
                    if self.cusp_method == 'enforce':
                        enforced_coeff = cusp_enforcing @ self.coeff[sp][:, i]
                        print('    constrained coefficients:', enforced_coeff[cusp_fixed_atorbs])
                        print('    after enforcing:         ', cusp_constraint @ enforced_coeff)
                        self.coeff[sp][:, i] = enforced_coeff
                if self.cusp_method != 'none':
                    constraint_violation = cusp_constraint @ self.coeff[sp][:, i]
                    assert np.all(np.abs(constraint_violation) < 1e-8)

        if self.do_plot_cusps:
            # Build a z-axis line through each atom from -0.5 to 0.5 (relative units)
            self.z = np.linspace(-0.5, 0.5, 500)
            # Create offset array of shape (3, self.z.size)
            offset = np.zeros((3, self.z.size))
            offset[2, :] = self.z
            # Create r with shape (self.sto.num_atom, 3, self.z.size)
            r = self.sto.atompos[:, :, None] + offset[None, :, :]
            self.val_pre = [
                [self.sto.eval_molorbs(r[atom], spin=sp)[:, self.fixed[sp]] for sp in range(self.Nspins)] for atom in range(self.sto.num_atom)
            ]
            self.lap_pre = [
                [self.sto.eval_molorb_derivs(r[atom], spin=sp)[2][:, self.fixed[sp]] for sp in range(self.Nspins)]
                for atom in range(self.sto.num_atom)
            ]

        self.sto.coeff = [c.T for c in self.coeff]
        self.sto.check_and_normalize()

        if self.do_plot_cusps:
            self.val_post = [
                [self.sto.eval_molorbs(r[atom], spin=sp)[:, self.fixed[sp]] for sp in range(self.Nspins)] for atom in range(self.sto.num_atom)
            ]
            self.lap_post = [
                [self.sto.eval_molorb_derivs(r[atom], spin=sp)[2][:, self.fixed[sp]] for sp in range(self.Nspins)]
                for atom in range(self.sto.num_atom)
            ]

        if self.cusp_method != 'none':
            print('Molorb values at nuclei after applying cusp constraint:')
            print(self.sto.eval_molorbs(self.sto.atompos.T))

        self.sto.writefile('stowfn.data')

    def plot_cusps(self):
        """Plot wavefunction values and local energies near each nucleus before and after cusp correction.

        Only runs when ``self.do_plot_cusps`` is True and
        ``apply_cusp_correction`` has stored the pre/post evaluation arrays.

        For each atom creates a 2-row subplot:
          * Top row: MO values ψ(z) along the z-axis through the nucleus,
            dashed = before correction, solid = after correction.
          * Bottom row: local kinetic energy E_loc = −∇²ψ/(2ψ) − Z/|r|,
            which should be flat at the nucleus after successful correction.

        Only orbitals that were flagged as cusp-violating (``self.fixed``) are
        plotted.  The y-axis of the local-energy panel is clipped to 1.5× the
        range of the corrected values to suppress the Coulomb singularity at
        r=0.

        Saves the figure to ``cusp_constraint.svg`` in the current directory.
        """
        if not self.do_plot_cusps:
            return
        import matplotlib.pyplot as plt

        # Create a 2 x Natom grid of subplots
        # Top row: wavefunction values (val)
        # Bottom row: local energies (eloc)
        fig, axes = plt.subplots(2, self.sto.num_atom, figsize=(4 * self.sto.num_atom, 4))
        # If only one atom, reshape axes into 2D array for consistency
        if self.sto.num_atom == 1:
            axes = np.array([axes]).reshape(2, 1)
        axval = [axes[0, at] for at in range(self.sto.num_atom)]
        axeloc = [axes[1, at] for at in range(self.sto.num_atom)]

        for at in range(self.sto.num_atom):
            eloc_min = 1e8
            eloc_max = -1e8
            for sp in range(self.Nspins):
                for i in range(np.sum(self.fixed[sp])):
                    vpre = self.val_pre[at][sp][:, i]  # wavefunction before correction
                    if np.allclose(vpre, 0):
                        print(f'val_pre at spin {sp}, orb {i}: is allclose to 0')
                        continue
                    vpost = self.val_post[at][sp][:, i]  # wavefunction after correction
                    sgn = np.sign(vpre[len(vpre) // 2])  # sign normalization
                    # Plot wavefunction before and after correction
                    (pl,) = axval[at].plot(self.z, sgn * vpre, '--')
                    axval[at].plot(self.z, sgn * vpost, '-', color=pl.get_color())
                    # Compute local energy before and after correction:
                    # E_loc = - (Laplacian / wavefunction) / 2 - Z / |r|
                    eloc_pre = -self.lap_pre[at][sp][:, i] / vpre / 2 - self.sto.atomnum[at] / np.abs(self.z)
                    eloc_post = -self.lap_post[at][sp][:, i] / vpost / 2 - self.sto.atomnum[at] / np.abs(self.z)
                    # Plot local energy before and after correction
                    axeloc[at].plot(self.z, eloc_pre, '--', color=pl.get_color())
                    axeloc[at].plot(self.z, eloc_post, '-', color=pl.get_color())
                    # Track min/max values for axis scaling
                    eloc_min = min(eloc_min, eloc_post[0], eloc_post[-1], eloc_post[len(eloc_post) // 2 - 1], eloc_post[len(eloc_post) // 2 + 1])
                    eloc_max = max(eloc_min, eloc_post[0], eloc_post[-1], eloc_post[len(eloc_post) // 2 - 1], eloc_post[len(eloc_post) // 2 + 1])
            # Expand y-limits around the middle value for better visualization
            eloc_mid = (eloc_min + eloc_max) / 2
            eloc_min = (eloc_min - eloc_mid) * 1.5 + eloc_mid
            eloc_max = (eloc_max - eloc_mid) * 1.5 + eloc_mid
            # Set axis ranges
            axval[at].set_xlim(self.z[0], self.z[-1])
            axeloc[at].set_xlim(self.z[0], self.z[-1])
            axeloc[at].set_ylim(eloc_min, eloc_max)
            # Add titles
            axval[at].set_title(f'Atom {at+1} (Z={self.sto.atomnum[at]}): Wavefunction')
            axeloc[at].set_title(f'Atom {at+1}: Local Energy')
        # Adjust layout and save figure
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
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Convert ADF TAPE21.asc to CASINO stowfn.data',
        epilog="""
        Examples:
          %(prog)s                              # use default: --cusp-method=enforce
          %(prog)s --plot-cusps                 # enable cusp plotting
          %(prog)s --cusp-method=enforce        # apply transformation to satisfy cusps (default)
          %(prog)s --cusp-method=project        # project out cusp-violating components
          %(prog)s --cusp-method=none           # disable any cusp correction
          %(prog)s --dump                       # generate a text dump of the parsed data
          %(prog)s --cart2harm-projection       # enforce pure spherical harmonics via projection
          %(prog)s --all-orbitals               # include also virtual orbitals (default: only occupied)
          %(prog)s --cusp-method=project --dump
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--plot-cusps', action='store_true', help='Enable plotting of nuclear cusps (e.g., density derivative at nuclei) (default: False)'
    )

    parser.add_argument(
        '--cusp-method',
        choices=['enforce', 'project', 'none'],
        default='enforce',
        help="""
            Choose how to handle nuclear cusp conditions:
            - enforce  : apply linear transformation to satisfy cusps (default)
            - project  : remove components that violate cusp conditions via projection
            - none     : do not apply any cusp correction
        """.strip(),
    )

    parser.add_argument(
        '--cart2harm-projection',
        action='store_true',
        help="""
                Enforce conversion from Cartesian to pure spherical harmonic Gaussian basis functions
                via orthogonal projection. Removes non-spherical components (e.g., s-type contamination
                in d/f shells like x²+y²+z²) that violate angular momentum purity. This ensures physical
                consistency but may change total energy if original orbitals contained such contamination.
            """.strip(),
    )

    parser.add_argument('--all-orbitals', action='store_true', default=False, help='If set, include also virtual orbitals (default: only occupied).')

    parser.add_argument('--dump', action='store_true', help='Generate a text dump (.txt) of the parsed ADF data for debugging (default: False)')

    args = parser.parse_args()

    adf_to_stowf = ADFToStoWF(args.plot_cusps, args.cusp_method, args.dump, args.cart2harm_projection, not args.all_orbitals)
    adf_to_stowf.run()


if __name__ == '__main__':
    main()
