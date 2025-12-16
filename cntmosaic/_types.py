"""
Shared type definitions used across multiple modules.

This module contains enum and type definitions that are used by multiple
packages to avoid circular imports.
"""

from enum import Enum


class StratMode(Enum):
    """
    Stratification configuration mode for a variable.

    Attributes
    ----------
    PARTIAL : str
        Information only available for participants (not contacts).
        Example: Participant gender recorded, contact gender unknown.
    FULL : str
        Information available for both participants and contacts.
        Example: Both participant and contact gender recorded.
    """

    PARTIAL = "partial"
    FULL = "full"
