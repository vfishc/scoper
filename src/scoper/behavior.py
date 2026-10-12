"""BEHAVIOR command line; usable without installing the AI2-THOR renderer."""
from scoper.backend_cli import main as backend_main


def main(argv=None):
    return backend_main('behavior', argv)


if __name__ == '__main__':
    raise SystemExit(main())
