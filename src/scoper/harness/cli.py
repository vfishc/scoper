import argparse
import importlib
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factory', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new artifact path')
    module, name = args.factory.split(':', 1)
    factory = getattr(importlib.import_module(module), name)
    runtime, adapter, start_kwargs = factory()
    try:
        runtime.start(**start_kwargs)
        result = runtime.run(adapter)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    finally:
        runtime.finish()
if __name__ == '__main__':
    main()
