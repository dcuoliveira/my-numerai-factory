"""
Load credentials from .env into the process environment.

Deliberately tiny and dependency-free -- this runs from cron, where a missing
python-dotenv would be a silent weekly failure.

.env is gitignored. Real environment variables always win over the file, so a
cron job or CI runner can override any value without editing anything on disk.
"""

import os

ENV_FILE = ".env"

REQUIRED = ("NUMERAI_PUBLIC_ID", "NUMERAI_SECRET_KEY")


def load_env(path=ENV_FILE, required=REQUIRED):
    """
    Read KEY=VALUE lines into os.environ. Returns the dict that was applied.

    :raises SystemExit: if a required key is missing after loading
    """
    applied = {}
    if os.path.exists(path):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip('"').strip("'")
                # An exported variable outranks the file.
                if key not in os.environ:
                    os.environ[key] = value
                applied[key] = value

    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise SystemExit(
            "missing credentials: %s\nSet them in %s or export them."
            % (", ".join(missing), path))
    return applied


def get_api():
    """A CryptoAPI authenticated from the environment."""
    from numerapi import CryptoAPI
    return CryptoAPI(public_id=os.environ["NUMERAI_PUBLIC_ID"],
                     secret_key=os.environ["NUMERAI_SECRET_KEY"])
