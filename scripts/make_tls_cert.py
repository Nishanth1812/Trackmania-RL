"""Generate the self-signed TLS credentials the TMRL server and worker pin (tlspyo file names).

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe scripts\\make_tls_cert.py --out secrets_local

Writes certificate.pem and key.pem. The directory is ignored by Git; never commit these files.
The hostname is "default", matching HOSTNAME in the TMRL config: the worker pins the certificate
rather than validating the changing Modal tunnel hostname.
"""
import argparse
import sys
from pathlib import Path


def generate(folder) -> None:
    """Write certificate.pem + key.pem (RSA-4096, self-signed, CN/SAN "default", 10 years).

    tlspyo's own generator fails with the installed pyOpenSSL, so build it with `cryptography`.
    """
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    folder = Path(folder)
    key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "default")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("default")]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    (folder / "certificate.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (folder / "key.pem").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="secrets_local")
    p.add_argument("--force", action="store_true", help="overwrite existing credentials")
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = [f for f in ("certificate.pem", "key.pem") if (out / f).exists()]
    if existing and not a.force:
        print(f"{existing} already exist in {out}; pass --force to replace them.", flush=True)
        return 1
    generate(out)
    print(f"Wrote {out / 'certificate.pem'} and {out / 'key.pem'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
