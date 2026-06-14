"""Compatibility alias for the renamed :mod:`randomwalks` package."""

import randomwalks as _randomwalks

__path__ = _randomwalks.__path__
__all__ = getattr(_randomwalks, "__all__", [])


def __getattr__(name):
    return getattr(_randomwalks, name)
