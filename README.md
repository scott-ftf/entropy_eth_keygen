# Entropy keygen

For experimenting with entropy sources. Can be used online to generate ethereum addresses from external sources, or offline using only local sources.

**⚠️ WARNING:** This script is **EXPERIMENTAL** for testing entropy sources. Use at **YOUR OWN RISK**.

Each entropy source includes Shannon calculations for insight into the randomness quality.

EXAMPLE OUTPUT:

```
Shannon entropy calculations

  Hardware RNG: 0.0
   Yubikey RNG: 7.148
Random Integer: 7.265
       Secrets: 7.239
    Dev Random: 7.221
    Time-based: 7.14
       Network: 7.178
        System: 7.234


Shannon Entropy Scale Guide
7.98 - 8.00 → Near-perfect randomness (Nuclear launch codes)
7.80 - 7.98 → Excellent randomness (Cryptographic apps)
7.50 - 7.80 → Strong randomness (Security-sensitive use)
7.00 - 7.50 → Moderate randomness (Acceptable but could be improved)
Below  7.00 → Weak randomness (not suitable, patterns likely)


FINAL COMBINED SHANNON ENTROPY: 7.987

```



At the end of the script, produces a private/public keypair and address

```
-----------BEGIN KEY OUTPUT-------------

private key:
0x61d2e91aaeda7a0996a319635a532c3476aa9a6f50cb8e6005e86f28db8debf4

public key:
0x95e3c44a3a90788361a77a04f9b072e010b756045a3d6c1787c104b42e1c838777e9a1c1120bd99b6fc82dae4c3db7a97c894c2f0481e2f3238da66109b73367

address:
0x627a512f29101d6366c2ce49f87866b5c6df3796

------------END KEY OUTPUT--------------

```



---



### Configure

Optionally uses <a href="https://api.random.org/">api.random.org</a> which generates true random numbers based on atmospheric noise as an entropy source

This source may be skipped, or to use it a free key must first be registered at https://accounts.random.org/create

Before launching the app, paste the key in the `random_org_api.secret` file in the root directory

---



### Install

Requires ubuntu LTS and python3 - uses all standard libraries, except for the `eth_keys` and `eth-hash` libraries, which are maintained by the ethereum foundation

```bash
pip3 install eth-hash[pycryptodome] eth-keys
```

Optionally uses a Yubikey as an Random Number Generator (RNG). This source may be skipped, or to use it, first install Yubicos key manager

``````bash
sudo snap install ykman
``````

---



### Run

In the root directory, type:

```bash
python3 entropy_ethereum_keygen.py
```

