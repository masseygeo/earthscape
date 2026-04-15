
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("earthscape")
    
except PackageNotFoundError:
    __version__ = "0.0"
