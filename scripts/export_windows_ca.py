from __future__ import annotations

import argparse
import ssl
from pathlib import Path

import certifi


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a PEM bundle from certifi and Windows trust stores.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    certificates = []
    for store in ("ROOT", "CA"):
        for certificate, encoding, _ in ssl.enum_certificates(store):
            if encoding == "x509_asn":
                certificates.append(ssl.DER_cert_to_PEM_cert(certificate))
    base = Path(certifi.where()).read_text(encoding="ascii")
    args.output.write_text(base + "\n" + "\n".join(dict.fromkeys(certificates)), encoding="ascii")
    print(f"Wrote {len(set(certificates))} Windows certificates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
