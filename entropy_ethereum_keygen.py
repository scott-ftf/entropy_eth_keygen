import hashlib
import secrets
import os
import requests
import time
import subprocess
import re
import binascii
import sys
import random
import psutil
import math
from eth_keys import keys

YOUR_API_KEY = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

def shannon_entropy(data):
    data_length = len(data)
    byte_counts = [data.count(byte) for byte in data]
    probabilities = [count / data_length for count in byte_counts]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    rounded_entropy = "{:.3f}".format(round(entropy, 3))
    return rounded_entropy 

def hex_viewer(data):
    hex_data = binascii.hexlify(data)
    hex_string = " ".join("{:02x}".format(c) for c in hex_data)
    ascii_string = "".join(chr(c) if 32 <= c <= 126 else "." for c in data)
    for i in range(0, len(hex_string), 48):
        hex_line = hex_string[i:i+48]
        ascii_line = ascii_string[i//3:i//3+16]
        print(hex_line, "  ", ascii_line)


def get_hardware_rng():
    print("\nHardware RNG generates entropy from hardware-level noise sources.")
    if not os.path.exists('/dev/hwrng'):
        print("Hardware RNG device not found on this system.")
        return b''
    if input("Do you want to use this source? (Y/n): ").strip().lower() in ['y', '']:
        with open('/dev/hwrng', 'rb') as f:
            entropy = f.read(32)
        hex_viewer(entropy)
        return entropy
    return b''

def get_random_integer():
    print("\nRandom Integer fetches a random number from random.org, an external service.")
    if input("Do you want to use this source? (Y/n): ").strip().lower() in ['y', '']:
        url = "https://api.random.org/json-rpc/4/invoke"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "jsonrpc": "2.0",
            "method": "generateIntegers",
            "params": {
                "apiKey": YOUR_API_KEY,
                "n": 256,
                "min": 1,
                "max": 0xFFFF
            },
            "id": 1
        }
        response = requests.post(url, json=payload, headers=headers)
        try:
            result = response.json()
            if 'result' in result:
                integers = result['result']['random']['data']
                entropy = b''.join(int.to_bytes(i, length=2, byteorder='big') for i in integers)
                entropy_hash = hashlib.sha256(entropy).digest()
                hex_viewer(entropy_hash)
                return entropy_hash
            else:
                print(f"Unexpected response: {result}")
                return b''
        except KeyError:
            print(f"Unexpected response: {result}")
            raise
    return b''

def get_time_based_entropy():
    print("\nTime-based entropy uses the current time in nanoseconds.")
    if input("Do you want to use this source? (Y/n): ").strip().lower() in ['y', '']:
        t = time.time_ns()
        entropy = int.to_bytes(t, byteorder='big', length=32)

        # Use the system's random number generator to add more entropy
        if os.path.exists('/dev/urandom'):
            with open('/dev/urandom', 'rb') as f:
                more_entropy = f.read(32)
            entropy = bytes([a ^ b for a, b in zip(entropy, more_entropy)])

        hex_viewer(entropy)
        return entropy
    
    return b''

def get_keyboard_entropy():
    print("\nKeyboard entropy allows you to provide randomness by typing random characters.")
    print("Type as much random text as you want, then press Enter...")
    random_text = input()
    entropy = random_text.encode()
    hex_viewer(entropy)
    return entropy

def get_secrets():
    print("\nSecrets library can generate cryptographically strong random numbers.")
    if input("Do you want to use this source? (Y/n): ").strip().lower() in ['y', '']:
        # Introduce random delay before generating the first token
        time.sleep(random.uniform(0.001, 0.1))  # Sleep for 1 to 100 milliseconds
        
        # Generate multiple tokens
        num_tokens = 5
        tokens = [secrets.token_bytes(32) for _ in range(num_tokens)]
        
        # XOR all the tokens together
        entropy = tokens[0]
        for token in tokens[1:]:
            entropy = bytes(b1 ^ b2 for b1, b2 in zip(entropy, token))
        
        # Introduce random delay after generating the tokens
        time.sleep(random.uniform(0.001, 0.1))  # Sleep for 1 to 100 milliseconds
        
        hex_viewer(entropy)
        return entropy
    
    return b''

def get_dev_random():
    print("\n/dev/random is a special file in Unix-like systems that serves as a random number generator.")
    if input("Do you want to use this source? (Y/n): ").strip().lower() in ['y', '']:
        # Introduce random delay before reading from /dev/random
        time.sleep(random.uniform(0.001, 0.1))  # Sleep for 1 to 100 milliseconds
        
        with open('/dev/random', 'rb') as f:
            entropy = f.read(32)
        
        # Introduce random delay after reading from /dev/random
        time.sleep(random.uniform(0.001, 0.1))  # Sleep for 1 to 100 milliseconds
        
        hex_viewer(entropy)
        return entropy
    return b''

def get_network_entropy():
    print("\nNetwork entropy fetches randomness from the network statistics on your machine.")
    if input("Do you want to use this source? (Y/n): ").strip().lower() in ['y', '']:
        # Collect data from network interfaces
        network_data = []
        for interface, stats in psutil.net_io_counters(pernic=True).items():
            data = f"{interface}{stats.bytes_sent}{stats.bytes_recv}{stats.packets_sent}{stats.packets_recv}"
            network_data.append(data.encode())

        # Collect timestamps from network-related events
        timestamp_data = []
        # Add code to collect timestamps from events (e.g., TCP/UDP connections, packets received)

        # Combine all collected data
        combined_data = b''.join(network_data) + b''.join(timestamp_data)

        # Hash the combined data
        entropy = hashlib.sha256(combined_data).digest()

        hex_viewer(entropy)
        return entropy
    return b''


def get_system_entropy():
    print("\nSystem entropy gathers system statistics such as CPU, memory usage, and buffer/cache.")
    if input("Do you want to use this source? (Y/n): ").strip().lower() in ['y', '']:
        output = subprocess.check_output(['top', '-b', '-n', '1'], universal_newlines=True).split("\n")
        entropy_data = []

        for line in output:
            # Add entropy from CPU statistics
            if "Cpu(s)" in line:
                cpu_values = re.findall(r'\d+', line)
                cpu_values = [value for value in cpu_values if not value.isdigit() or '.' not in value]
                entropy_data.extend(cpu_values)

            # Add entropy from memory statistics
            elif "KiB Mem" in line:
                memory_values = re.findall(r'\d+', line)
                memory_values = [value for value in memory_values if not value.isdigit() or '.' not in value]
                entropy_data.extend(memory_values)

            # Add entropy from buffer/cache statistics
            elif "KiB Buff" in line:
                buffer_cache_values = re.findall(r'\d+', line)
                buffer_cache_values = [value for value in buffer_cache_values if not value.isdigit() or '.' not in value]
                entropy_data.extend(buffer_cache_values)

            # Add entropy from task statistics
            elif "Tasks:" in line:
                tasks_values = re.findall(r'\d+', line)
                tasks_values = [value for value in tasks_values if not value.isdigit() or '.' not in value]
                entropy_data.extend(tasks_values)

            # Add entropy from interrupts, context switches, and CPU states
            elif "Csw" in line or "intr" in line or "CPU" in line:
                values = re.findall(r'\d+', line)
                values = [value for value in values if not value.isdigit() or '.' not in value]
                entropy_data.extend(values)

        # Combine the entropy from different sources
        entropy = "".join(entropy_data).encode()

        # Use a cryptographic hash function to extract a fixed-size (32-byte) output
        entropy_hash = hashlib.sha256(entropy).digest()

        hex_viewer(entropy_hash)
        return entropy_hash

    return b''

def generate_entropy():
    entropy_sources = [
        ("      Keyboard", get_keyboard_entropy()),
        ("  Hardware RNG", get_hardware_rng()),
        ("Random Integer", get_random_integer()),
        ("       Secrets", get_secrets()),
        ("    Dev Random", get_dev_random()),
        ("    Time-based", get_time_based_entropy()),
        ("       Network", get_network_entropy()),
        ("        System", get_system_entropy())
    ]

    combined_sources = b''.join(entropy for _, entropy in entropy_sources)
    final_entropy_shannon = shannon_entropy(combined_sources)

    print(f"\nGenerating final entropy")
    hex_viewer(combined_sources)

    print(f"\n\nShannon entropy calculations\n")
    for source_name, entropy in entropy_sources:
        entropy_shannon = shannon_entropy(entropy)
        print(f"{source_name}: {entropy_shannon}")

    print(f"\nFinal Combined Entropy Shannon Entropy: {final_entropy_shannon}")
    return hashlib.sha256(combined_sources).digest()

def create_ethereum_keys(entropy):
    private_key_int = int.from_bytes(entropy, byteorder='big')
    private_key = keys.PrivateKey(private_key_int.to_bytes(32, byteorder='big'))
    public_key = private_key.public_key
    eth_address = public_key.to_address()

    return private_key, public_key, eth_address

# dont just clear screen, but dump screen buffer    
def clearScreen():
    print("\n\n     This is the end of this script") 
    print("      record key outputs securely")  
    print("\nHit ENTER to clear screen buffer and exit")  
    input()
    os.system("clear && printf '\e[3J'")
    sys.exit()


if __name__ == '__main__':
    entropy = generate_entropy()
    
    print("\nseeding key generation...")
    private_key, public_key, eth_address = create_ethereum_keys(entropy)

    print("\n\n-----------BEGIN KEY OUTPUT-------------")
    print("\n\nprivate key:")
    print(private_key)
    print("\npublic key:")
    print(public_key)
    print("\naddress:")
    print(eth_address)
    print("\n\n------------END KEY OUTPUT--------------")
    
# clear screen on user input
    clearScreen()
