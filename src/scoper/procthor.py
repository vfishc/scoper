"""ProcTHOR command line; usable without installing OmniGibson."""
from scoper.backend_cli import main as backend_main


def main(argv=None):
    return backend_main('procthor', argv)


if __name__ == '__main__':
    raise SystemExit(main())
