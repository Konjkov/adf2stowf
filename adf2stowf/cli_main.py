import argparse


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

    return args.plot_cusps, args.cusp_method, args.dump, args.cart2harm_projection, not args.all_orbitals
