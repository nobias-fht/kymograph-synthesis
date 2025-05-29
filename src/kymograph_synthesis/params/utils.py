import subprocess


def find_version() -> str:
    # Will be version or commit hash depending if a proper version is found
    # For now since there are no releases only commit hash is saved.
    return _find_commit_hash()


def _find_commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
