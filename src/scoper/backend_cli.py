"""Backend-scoped entrypoints sharing only argument routing and configuration."""
import argparse
from pathlib import Path
from scoper import cli


def main(backend, argv=None):
    parser = argparse.ArgumentParser(prog='scoper-' + backend)
    parser.add_argument('--config', type=Path)
    parser.add_argument('command', choices=cli.COMMANDS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if any(x == '--backend' or x.startswith('--backend=') for x in args.arguments):
        parser.error('This entrypoint already selects ' + backend + '; omit --backend')
    routed = [] if args.config is None else ['--config', str(args.config)]
    routed.append(args.command)
    if args.command in {'generate', 'oracle', 'evaluate', 'doctor'}:
        routed += ['--backend', backend]
    routed += args.arguments
    return cli.main(routed)
