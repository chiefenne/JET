"""Mesh generation for transformed coordinates."""

import logging

import numpy as np

from .config import MeshConfig

logger = logging.getLogger(__name__)


class Mesh:
    """2D rectangular transformed grid (gsi, eta)."""

    def __init__(self, config: MeshConfig):
        self.dgsi = config.dgsi
        self.etae = config.etae
        self.stretch = config.stretch

        # number of points in eta direction
        if config.stretch < 1.0001:
            etamax = config.etae / config.deta1
        else:
            # equation (13.49, page 400) Cebeci & Bradshaw
            etamax = (np.log(1.0 + (config.stretch - 1.0) * config.etae /
                             config.deta1) / np.log(config.stretch))

        self.etamax = int(etamax)
        self.gsimax = config.gsimax + 1

        # build gsi array
        self.gsi = np.zeros(self.gsimax + 1)
        self.gsi[0] = 1.0
        for i in range(1, self.gsimax + 1):
            self.gsi[i] = self.gsi[i - 1] + config.dgsi

        # build eta arrays
        self.eta = np.zeros(self.etamax)
        self.deta = np.zeros(self.etamax - 1)
        self.a = np.zeros(self.etamax)

        self.deta[0] = config.deta1
        for j in range(1, self.etamax - 1):
            self.deta[j] = config.stretch * self.deta[j - 1]
        for j in range(1, self.etamax):
            self.a[j] = 0.5 * self.deta[j - 1]
            self.eta[j] = self.eta[j - 1] + self.deta[j - 1]

        logger.info('MESH PROPERTIES')
        logger.info('  GSI start: %s', self.gsi[0])
        logger.info('  GSI spacing: %s', config.dgsi)
        logger.info('  ETA spacing (initial): %s', config.deta1)
        logger.info('  ETA growth rate: %s', config.stretch)
        logger.info('  ETA boundary (given): %s', config.etae)
        logger.info('  ETA boundary (calculated): %s', self.eta[-1])
        logger.info('  Grid points GSI: %d', self.gsimax)
        logger.info('  Grid points ETA: %d', self.etamax)
