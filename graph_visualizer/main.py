"""Entry point for the sparse thermal graph visualizer."""

from __future__ import annotations

from .app import GraphVisualizerApp


def main() -> None:
    GraphVisualizerApp().show()


if __name__ == "__main__":
    main()
