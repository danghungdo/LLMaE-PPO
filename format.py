#!/usr/bin/env python3
"""Format and lint the codebase."""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"Running {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"{description} failed:")
        print(result.stdout)
        print(result.stderr)
        return False
    else:
        print(f"{description} passed")
        if result.stdout.strip():
            print(result.stdout)
        return True


def main():
    """Main formatting function."""
    files = "llmae_ppo/ main.py evaluate.py"
    success = True

    print("Formatting and linting codebase...")
    print("=" * 50)

    # Format with black
    success &= run_command(f"black {files}", "Black formatting")

    # Sort imports with isort
    success &= run_command(f"isort {files}", "Import sorting")

    # Lint with flake8
    success &= run_command(f"flake8 {files}", "Flake8 linting")

    print("=" * 50)
    if success:
        print("All formatting and linting passed!")
        print("\nYour code is now properly formatted and linted.")
    else:
        print("Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
