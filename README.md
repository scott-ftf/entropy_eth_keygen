# Entropy keygen

For experimenting with entropy sources. Can also be used online (or offline) to generate ethereum addresses.

Optionaly uses <a href="https://api.random.org/">api.random.org</a> as an entropy source, which requires a free api key

requires ubuntu LTS and python3- uses all standard libraries, except for the eth_keys library from the ethereum foundation

sudo is required for running this script to access hardware sources of addtional entropy, so eth keys must be installed with:
```bash
sudo pip install eth-hash[pycryptodome] eth-keys
```

