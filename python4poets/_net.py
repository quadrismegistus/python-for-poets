"""Opening web addresses safely, on any computer.

Python checks a website's security certificate against a list of trusted
certificate authorities. Python from python.org on a Mac comes without that
list (until you run "Install Certificates.command"), so every download fails
with CERTIFICATE_VERIFY_FAILED. We use the list from the `certifi` package
instead, which pip installs along with this one.
"""

import contextlib
import ssl
import urllib.request


def ssl_context():
    """An SSL context that trusts certifi's certificate authorities (or the system's, without certifi)."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def urlopen(url, timeout=60):
    """urllib.request.urlopen, checking certificates against certifi's list."""
    return urllib.request.urlopen(url, timeout=timeout, context=ssl_context())



@contextlib.contextmanager
def trusted_certificates():
    """
    While inside this block, other libraries' downloads (e.g. nltk.download,
    which can't be given an SSL context) also use certifi's list.
    """
    original = ssl._create_default_https_context
    ssl._create_default_https_context = ssl_context
    try:
        yield
    finally:
        ssl._create_default_https_context = original
