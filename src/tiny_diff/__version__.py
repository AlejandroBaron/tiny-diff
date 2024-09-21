import os
import subprocess

_PACKAGE_ROOT = os.path.dirname(__file__)
_VERSION_PATH = os.path.join(os.path.dirname(_PACKAGE_ROOT), "version.info")

def get_version_from_git():
    """Get the version using `git describe --tags` if available."""
    try:
        version = subprocess.run(
            ["git", "describe", "--tags"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        return version
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Return None if the command fails or git is not available
        return None

# Default fallback version
fallback_version = "0.0.0"

try:
    if os.path.exists(_VERSION_PATH):
        with open(_VERSION_PATH, encoding="utf-8") as fo:
            version = fo.readlines()[0].strip()
    else:
        # Fallback to git if version.info is not available
        version = get_version_from_git() or fallback_version
except FileNotFoundError:
    version = fallback_version

__version__ = version
