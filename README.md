# Entropy keygen

For experimenting with entropy sources. Can be used online to generate Ethereum addresses from external sources, or offline using only local sources.

Following the entropy generation, the script provides a series of tests to evaluative aspects of the randomness

| Test Name                           | Purpose                                   |
| ----------------------------------- | ----------------------------------------- |
| Shannon Entropy Calculation         | Measures overall unpredictability of data |
| Chi-Square Goodness-of-Fit Test     | Checks for uniform byte distribution      |
| Serial Correlation Coefficient Test | Detects repeating patterns in data        |
| Monte Carlo Estimation of π Test    | Tests statistical randomness using π      |
| Data Compression Ratio Analysis     | Estimates redundancy and compressibility  |
| Bit Frequency (Monobit) Test        | Ensures an even mix of 0s and 1s          |



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



**⚠️ WARNING:** This script is **EXPERIMENTAL** for testing entropy sources. Use at **YOUR OWN RISK**.



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

