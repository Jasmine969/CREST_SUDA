"""
Transform ringID, neuronID, SIP-ID to axial position and the reverses.
The index is 0-based
Default unit: mm
"""
import numpy as np


def ringID2x(identifier):
    return 0.2 * np.array(identifier)


def x2ringID(x):
    return np.array(x) * 5


def SIP_ID2x(identifier):
    return 0.2 * (np.array(identifier) + 8)


def x2SIP_ID(x):
    return np.array(x) * 5 - 8


def neuronID2x(identifier):
    return 0.2 * (2 * np.array(identifier) + 9)


def x2neuronID(x):
    return (np.array(x) * 5 - 9) * 0.5
