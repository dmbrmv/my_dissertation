#!/usr/bin/env python
"""Export a Marimo notebook to static HTML for GitHub Pages.

Usage:
    pixi run python scripts/export_notebook.py notebooks/c1_0_GeoClusters.py

The notebook will be exported to docs/notebooks/{name}.html with all outputs included.
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/export_notebook.py <notebook.py>")
        print("Example: python scripts/export_notebook.py notebooks/c1_0_GeoClusters.py")
        sys.exit(1)

    notebook_path = Path(sys.argv[1])
    if not notebook_path.exists():
        print(f"Error: {notebook_path} does not exist")
        sys.exit(1)

    # Create output directory
    output_dir = Path("docs/notebooks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to HTML
    output_file = output_dir / f"{notebook_path.stem}.html"
    print(f"Exporting {notebook_path} -> {output_file}")

    result = subprocess.run(
        ["marimo", "export", "html", str(notebook_path), "-o", str(output_file), "--include-code"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    print(f"Successfully exported to {output_file}")
    print(f"Commit this file and push to deploy to GitHub Pages")


if __name__ == "__main__":
    main()
