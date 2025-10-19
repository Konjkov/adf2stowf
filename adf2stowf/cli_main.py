import argparse


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Convert ADF TAPE21.asc to CASINO stowfn.data',
        epilog="""
        Examples:
          %(prog)s                         # use defaults: PLOTCUSPS=False, CUSP_ENFORCE=True
          %(prog)s --plot-cusps            # set PLOTCUSPS=True
          %(prog)s --no-cusp-enforce       # set CUSP_ENFORCE=False
          %(prog)s --plot-cusps --no-cusp-enforce
          %(prog)s --dump                  # generate a text dump of the parsed data
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--plot-cusps', action='store_true', help='Enable plotting of cusps (default: False)')

    parser.add_argument('--no-cusp-enforce', dest='cusp_enforce', action='store_false', help='Disable cusp enforcement (default: True)')

    parser.add_argument('--dump', action='store_true', help='Generate a text dump file (.txt) of the parsed ADF data (default: False)')

    args = parser.parse_args()
    return args.plot_cusps, args.cusp_enforce, args.dump
