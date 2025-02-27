

# Entropy keygen

For experimenting with entropy sources. The entropy can  then be used to generate Ethereum seed-phrase, private key and address.

The main intend for this script is experimenting with entropy sources. Following the entropy generation, the script provides a series of tests to evaluative aspects of the randomness

| Test Name                           | Purpose                                   |
| ----------------------------------- | ----------------------------------------- |
| Shannon Entropy Calculation         | Measures overall unpredictability of data |
| Chi-Square Goodness-of-Fit Test     | Checks for uniform byte distribution      |
| Serial Correlation Coefficient Test | Detects repeating patterns in data        |
| Monte Carlo Estimation of π Test    | Tests statistical randomness using π      |
| Data Compression Ratio Analysis     | Estimates redundancy and compressibility  |
| Bit Frequency (Monobit) Test        | Ensures an even mix of 0s and 1s          |



At the end of the script, 128 or 256 bits of the generated entropy is used to generate a 12 or 24 word mnemonic phrase using bitcoin's wordlist from https://raw.githubusercontent.com/bitcoin/bips/master/bip-0039/english.txt, adds a checksum for improved integrity, and converts back to a to derive a private key, and Ethereum payment address

```
-----------BEGIN KEY OUTPUT-------------
Seed Phrase:
expose mammal polar museum addict box ordinary scene creek chair jealous pyramid twist narrow ancient embody where also prepare world tag lesson verb olympic

private key:
ea19bdf326dc3d926ff996c8be91001213216122a46c49255b33431b6a3222de

public key:
0x6946270E4EDd13c5d0b4A799daA69c7743E6F8f9
------------END KEY OUTPUT--------------


```



---

**⚠️ WARNING:** This script is **EXPERIMENTAL** for testing entropy sources. Use at **YOUR OWN RISK**.

---



### Configure

Optionally uses <a href="https://api.random.org/">api.random.org</a> which generates true random numbers based on atmospheric noise as an entropy source

This source may be skipped, or to use it a free key must first be registered at https://accounts.random.org/create

Before launching the app, paste the key in the `random_org_api.secret` file in the root directory

---



### Install

Requires ubuntu LTS and python3 - uses all standard libraries, except for the `eth_keys`, `eth-account`, `eth-utils` and `eth-hash` libraries, which are maintained by the Ethereum foundation. Install the required libraries from the repo root with: 

```bash
pip install -r requirements.txt
```

Optionally uses a Yubikey as an Random Number Generator (RNG). This source may be skipped, or to use it, first install Yubico's key manager

``````bash
sudo snap install ykman
``````

---



### Run

In the root directory, type:

```bash
python3 entropy_ethereum_keygen.py
```

