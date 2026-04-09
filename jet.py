#!/usr/bin/env python
"""2D Turbulent Heated Free Jet - Entry Point.

Usage:
    python jet.py
    python jet.py config.ini
"""

import sys

from jet import Simulation
from jet.config_loader import load_simulation_config


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.ini"
    config = load_simulation_config(config_path)

    sim = Simulation(config)
    sim.run()
    sim.save()
    sim.plot()


if __name__ == '__main__':
    main()
