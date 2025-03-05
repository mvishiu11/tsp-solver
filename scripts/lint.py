#!/usr/bin/env python
"""
Script to run all linters: isort, black, ruff, and flake8 on specified directories.
"""
import subprocess
import sys


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["."]

    try:
        print("[Lint] Running isort...")
        subprocess.run(["isort"] + targets, check=True)

        print("[Lint] Running black...")
        subprocess.run(["black"] + targets, check=True)

        print("[Lint] Running ruff...")
        subprocess.run(["ruff", "check", "--fix"] + targets, check=True)

        print("[Lint] Running flake8...")
        subprocess.run(["flake8"] + targets, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Linting failed: {e}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
