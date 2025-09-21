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

    parser.add_argument('--dump', action='store_true', help='Generate a text dump (.txt) of the parsed ADF data for debugging (default: False)')

    args = parser.parse_args()

    return args.plot_cusps, args.cusp_method, args.dump
