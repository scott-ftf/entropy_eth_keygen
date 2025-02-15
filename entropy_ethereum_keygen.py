import hashlib
import secrets
import os
import json
import urllib.request
import time
import subprocess
import binascii
import sys
import random
import struct
import hashlib
import shutil
import math
from eth_keys import keys
import getpass
from collections import Counter
import socket
import re

# Define the API key file in the application root directory
API_KEY_FILE = os.path.join(os.path.dirname(__file__), "random_org_api.secret")


def shannon_entropy(data):
    """Calculate Shannon entropy of a byte sequence."""
    if not data:
        return 0.0  # Avoid division by zero if data is empty

    data_length = len(data)
    counter = Counter(data)  # Count occurrences of each byte (0-255)
    
    probabilities = [count / data_length for count in counter.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

    return round(entropy, 3)  


def hex_viewer(data, shannon=True):
    """Displays data in hexadecimal format with ASCII representation, and calculates entropy."""
    
    if not data:
        print("\nNo data provided for hex viewer.")
        return

    hex_data = binascii.hexlify(data)
    hex_string = " ".join("{:02x}".format(c) for c in hex_data)
    ascii_string = "".join(chr(c) if 32 <= c <= 126 else "." for c in data)

    for i in range(0, len(hex_string), 48):
        hex_line = hex_string[i:i+48]
        ascii_line = ascii_string[i//3:i//3+16]
        print(hex_line, "  ", ascii_line)

    if shannon:
        counter = Counter(data)
        print(f"\nData Length: {len(data)} bytes")
        print(f"Unique Byte Count: {len(counter)}")
        print("Entropy Shannon:", shannon_entropy(data))  


def run_with_sudo(command):
    """Run a command with sudo privileges"""
    password = getpass.getpass("Enter sudo password: ")

    try:
        # Run the command with sudo, ensuring we capture only the output
        process = subprocess.Popen(
            f"sudo -S {command}",
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        output, error = process.communicate(password + "\n")

        if process.returncode != 0:
            print(f"Failed to execute sudo command: {error.strip()}")
            return None

        return output.strip()  # Remove extra newlines/spaces
    except Exception as e:
        print(f"Error running sudo command: {e}")
        return None


def get_hardware_rng():
    print("\nHardware RNG (if exists, may require sudo)")

    if not os.path.exists('/dev/hwrng'):
        print("No hardware RNG device found. Skipping...")
        return b''

    if input("Do you want to try this source? (y/N): ").strip().lower() in ['n', '']:
        print("Skipping hardware RNG check.")
        return b''

    try:
        # Try reading without sudo
        with open('/dev/hwrng', 'rb') as f:
            entropy = f.read(256)  # 🔹 Increased to 256 bytes for better entropy
        print("Successfully read from hardware RNG (without sudo).")
        hex_viewer(entropy)
        return entropy

    except PermissionError:
        print("Permission denied. You may need sudo access to read from the hardware RNG.")

        # Ask the user if they want to retry with sudo or skip
        choice = input("Would you like to enter sudo to try again? (y/N): ").strip().lower()
        if choice in ['n', '']:
            print("Skipping hardware RNG check.")
            return b''

        # Use sudo to check if the hardware RNG can be accessed
        command = ["sudo", "dd", "if=/dev/hwrng", "bs=256", "count=1", "status=none"]
        try:
            process = subprocess.run(command, capture_output=True, text=False)  # Avoids password handling

            if process.returncode != 0:
                print(f"Failed to read from hardware RNG with sudo: {process.stderr.decode().strip()}")
                print("Skipping hardware RNG check.")
                return b''

            entropy = process.stdout  # No need to re-encode
            print("Successfully read from hardware RNG (with sudo).")
            hex_viewer(entropy)
            return entropy

        except Exception as e:
            print(f"Error using sudo for hardware RNG: {e}")
            print("Skipping hardware RNG check.")
            return b''


def install_ykman():
    """Check if ykman is installed; if not, ask for permission and install via Snap."""
    if shutil.which("ykman") is None:
        print("\nYubico `ykman` is required to use a YubiKey for entropy.")
        choice = input("Would you like to install `ykman` now? (Y/n): ").strip().lower()
        if choice not in ['y', '']:
            print("Skipping YubiKey RNG extraction.")
            return False
        
        print("\nInstalling `ykman` via Snap...")
        try:
            subprocess.run(["sudo", "snap", "install", "ykman"], check=True)
            print("`ykman` installed successfully.")
            return True
        except subprocess.CalledProcessError:
            print("Installation of `ykman` failed. Skipping YubiKey RNG extraction.")
            return False
    return True


def check_yubikey_status():
    """Check if a YubiKey is detected and OpenPGP is enabled"""
    try:
        output = subprocess.check_output(
            ["ykman", "info"],
            universal_newlines=True,
            stderr=subprocess.DEVNULL  # Suppress warnings and errors
        )
        if "OpenPGP" in output:
            print("YubiKey detected with OpenPGP enabled.")
            return True
        else:
            print("YubiKey detected, but OpenPGP is not enabled.")
            return False
    except subprocess.CalledProcessError:
        print("Failed to detect YubiKey using `ykman info`. Ensure it is plugged in.")
        return False


def get_yubikey_rng():
    """Attempt to extract entropy from YubiKey using GnuPG."""
    print("\nYubiKey has a built-in random number generator via OpenPGP applet")
    choice = input("Do you want to use this source? (Y/n): ").strip().lower()
    if choice not in ['y', '']:
        print("Skipping YubiKey RNG extraction.")
        return b''

    input("\nPlug in your YubiKey and press Enter when ready...")

    if not install_ykman():
        return b''

    if not check_yubikey_status():
        print("Skipping YubiKey RNG extraction.")
        return b''

    print("\nAttempting to extract entropy from YubiKey via GPG...")

    entropy = b''
    try:
        # Generate 8 blocks of 32 bytes each (for 256 bytes total)
        for _ in range(8):  
            entropy += subprocess.check_output(["gpg", "--gen-random", "0", "32"], universal_newlines=False)

        if len(entropy) != 256:
            print("Unexpected entropy size! Received:", len(entropy), "bytes")

        print("Successfully extracted 256 bytes of entropy from YubiKey.")
        hex_viewer(entropy)
        return entropy
    except subprocess.CalledProcessError:
        print("Failed to extract entropy from YubiKey using GPG.")
        return b''


def load_api_key():
    api_key_path = "random_org_api.secret"

    if not os.path.exists(api_key_path):
        print(f"\nNo valid API key found in {os.path.abspath(api_key_path)}")
        print("Ensure the key is correctly formatted and not inside comments.")
        print("See the above file for instructions on registering a free key to use this source.\n")
        input("Press ANY KEY to skip for now...")  # Wait for user input
        return None

    with open(api_key_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("#") or not line:  # Skip comments and empty lines
                continue

            # Check if the key matches the expected UUID format
            key_pattern = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
            if key_pattern.fullmatch(line):
                return line

    print(f"\nNo valid API key found in {os.path.abspath(api_key_path)}")
    print("Ensure the key is correctly formatted and not inside comments.")
    print("See the above file for instructions on registering a free key to use this source.\n")
    input("Press ANY KEY to skip for now...")  # Wait for user input
    return None


def get_random_integer():
    print("\nRandom.org generates true random numbers based on atmospheric noise (requires free API key in config)")   
    if input("Do you want to use this source? (Y/n): ").strip().lower() not in ['y', '']:
        print("Skipping Random.org entropy extraction.")
        return b''

    url = "https://api.random.org/json-rpc/4/invoke"
    headers = {'Content-Type': 'application/json'}
    api_key = load_api_key()
    if not api_key:
        return b''  # Exit gracefully if no valid API key
    
    payload = {
        "jsonrpc": "2.0",
        "method": "generateIntegers",
        "params": {
            "apiKey": api_key,
            "n": 128,  # Request 128 integers (each 2 bytes, total 256 bytes)
            "min": 0,
            "max": 0xFFFF,
            "replacement": True  # Ensure truly independent values
        },
        "id": 1
    }

    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())

        if result.get('result') and result['result'].get('random') and result['result']['random'].get('data'):
            integers = result['result']['random']['data']
            entropy = b''.join(i.to_bytes(2, 'big') for i in integers)

            if len(entropy) == 256:
                print("Successfully retrieved 256 bytes of entropy from Random.org.")
            else:
                print(f"Warning: Received {len(entropy)} bytes instead of 256!")

            hex_viewer(entropy)
            return entropy

        else:
            print(f"Unexpected response format: {result}")
            return b''

    except urllib.error.URLError as e:
        print(f"Failed to fetch from Random.org: {e}")
        return b''


def trim_zeros(data):
    """Remove only leading and trailing zeros, not internal ones."""
    return data.lstrip(b'\x00').rstrip(b'\x00')


def get_time_based_entropy():
    print("\nTime-based entropy with user-assisted randomness.")

    if input("Do you want to use this source? (Y/n): ").strip().lower() in ['n']:
        print("Skipping time-based entropy.")
        return b''

    timestamps = []
    print("\nPress Enter **six times** to capture timing randomness.\n")

    # Capture user-driven time variations
    for i in range(6):
        input(f"Press Enter to randomly select a timestamp ({i+1}/6)...")
        ts = time.time_ns()
        mono_ts = time.monotonic_ns()
        perf_ts = time.perf_counter_ns()
        
        # Store multiple time sources
        timestamps.append((ts, mono_ts, perf_ts))

    # Compute time deltas between user presses
    intervals = [
        (timestamps[i+1][0] - timestamps[i][0]) ^  # Normal time delta
        (timestamps[i+1][1] - timestamps[i][1]) ^  # Monotonic delta
        (timestamps[i+1][2] - timestamps[i][2])    # Perf counter delta
        for i in range(5)
    ]

    # XOR all timestamps and intervals together to seed entropy
    mixed_time = timestamps[0][0]
    for i in range(1, len(timestamps)):
        mixed_time ^= timestamps[i][0] ^ timestamps[i][1] ^ timestamps[i][2]
    for interval in intervals:
        mixed_time ^= interval

    entropy_bytes = bytearray()
    prev_time = mixed_time  # Seed automated loop with user-generated randomness

    while len(entropy_bytes) < 256:  # Ensure 256 bytes of entropy
        # Capture system timing variations in each loop
        current_time = time.time_ns()
        monotonic_time = time.monotonic_ns()
        perf_time = time.perf_counter_ns()
        
        # Mix in collected user timing randomness
        prev_time ^= (current_time ^ monotonic_time ^ perf_time)
        
        # Shuffle bits before selecting bytes
        prev_time = ((prev_time >> 3) ^ (prev_time << 7)) & 0xFFFFFFFFFFFFFFFF
        
        # Convert to bytes
        entropy_chunk = prev_time.to_bytes(8, byteorder='little')

        # Append all bytes, including zeros (zeros may be useful for randomness)
        entropy_bytes.extend(entropy_chunk)

        # Use user-collected timing deltas to influence loop timing
        delay_index = random.randint(0, len(intervals) - 1)
        delay = abs(intervals[delay_index]) % 100000  # Delay based on user timing
        target_time = time.perf_counter_ns() + delay
        while time.perf_counter_ns() < target_time:
            pass  # Allow system to naturally introduce more jitter

    entropy_bytes = entropy_bytes[:256]
    print(f"\nGenerated {len(entropy_bytes)} bytes of entropy from time-based and user-driven randomness.")
    hex_viewer(entropy_bytes)

    return entropy_bytes  

'''
REMOVE - human biased, low unique bytes, low shannon compared to otehr sourcess

def get_keyboard_entropy():
    print("\nKeyboard entropy allows you to provide randomness by typing random characters.")
    if input("Do you want to use this source? (Y/n): ").strip().lower() in ['n']:
        print("Skipping keyboard entropy.")
        return b''

    print("Type AT LEAST 256 characters of random text, then press Enter...")

    typed_data = []
    timing_data = []
    prev_time = time.time()

    while len(typed_data) < 256:  
        user_input = input("\n> ")

        if not user_input:
            print("You must enter at least one character.")
            continue

        typed_data.extend(user_input)

        # Capture keystroke timing deltas for extra entropy
        curr_time = time.time()
        time_diff = int((curr_time - prev_time) * 1e6)  # Convert to microseconds
        prev_time = curr_time
        timing_data.append(chr(time_diff % 256))  

        # Show remaining characters after user presses Enter
        remaining_chars = 256 - len(typed_data)
        if remaining_chars > 0:
            print(f"You need at least {remaining_chars} more characters.")

    # Combine typed characters with keystroke timing data
    entropy = "".join(typed_data + timing_data).encode()

    print("\nSufficient keyboard randomness collected.")
    hex_viewer(entropy)
    return entropy
'''

def get_secrets():
    print("\nSecrets library can generate cryptographically strong random numbers.")
    
    if input("Do you want to use this source? (Y/n): ").strip().lower() in ['y', '']:
        # Generate 256 bytes directly from secrets
        entropy = secrets.token_bytes(256)

        hex_viewer(entropy)
        return entropy

    return b''


def get_dev_random():
    print("\n/dev/random is a special file in Unix-like systems that serves as a random number generator.")    
    if input("Do you want to use this source? (Y/n): ").strip().lower() in ['y', '']:
        with open('/dev/random', 'rb') as f:
            entropy = f.read(256)
        
        hex_viewer(entropy)
        return entropy

    return b''


def get_network_entropy():
    print("\nNetwork entropy fetches randomness from network statistics.")
    
    if input("Do you want to use this source? (Y/n): ").strip().lower() not in ['y', '']:
        print("Skipping network entropy.")
        return b''

    network_data = bytearray()

    def collect_network_metrics():
        """Gathers multiple sources of network randomness using standard libraries."""
        temp_data = bytearray()
        snapshot_count = 5  # Number of network snapshots

        # Capture multiple snapshots over time
        history = []
        for _ in range(snapshot_count):
            snapshot = bytearray()

            # Collect active network connections
            try:
                connections = socket.getaddrinfo(None, None, proto=socket.IPPROTO_TCP)
                for conn in connections:
                    snapshot.extend(struct.pack("!H", conn[4][1]))  # Port number as 2 bytes
            except Exception:
                pass  # Ignore errors

            # Collect system hostname & IP addresses
            try:
                hostname = socket.gethostname().encode()
                snapshot.extend(hostname)
                
                ip_addr = socket.gethostbyname(hostname)
                snapshot.extend(socket.inet_aton(ip_addr))
            except Exception:
                pass  # Ignore errors

            # Collect random MAC addresses using interface names
            try:
                for iface in os.listdir('/sys/class/net/'):
                    mac_path = f"/sys/class/net/{iface}/address"
                    if os.path.exists(mac_path):
                        with open(mac_path, 'r') as f:
                            mac = f.read().strip().replace(":", "")
                            snapshot.extend(bytes.fromhex(mac))
            except Exception:
                pass  # Ignore errors

            # Collect routing table (Linux-specific)
            try:
                with open("/proc/net/route", "r") as f:
                    routes = f.read().encode()
                    snapshot.extend(routes[:64])  # Limit to avoid excessive data
            except Exception:
                pass  # Ignore errors

            # Capture timestamps
            timestamp1 = time.time_ns()
            timestamp2 = time.monotonic_ns()
            snapshot.extend(timestamp1.to_bytes(8, byteorder='big'))
            snapshot.extend(timestamp2.to_bytes(8, byteorder='big'))

            # Add to history
            history.append(snapshot)

            # Small random delay between samples
            time.sleep((time.time_ns() % 50000) / 1e9)  # Microsecond-based delay

        # Flatten history into temp_data
        temp_data.extend(b''.join(history))

        return temp_data

    # Ensure at least 256 bytes are collected
    while len(network_data) < 256:
        new_data = collect_network_metrics()
        network_data.extend(new_data)
        if len(new_data) == 0:  # Prevent infinite loops
            print("Warning: No new network data collected, stopping early.")
            break  

    # Trim to exactly 256 bytes
    network_data = network_data[:256]

    # Final SHA-512 hash to distribute randomness
    final_entropy = hashlib.shake_256(network_data).digest(256)
    hex_viewer(final_entropy)

    return final_entropy

   
def get_system_entropy():
    print("\nSystem entropy gathers CPU, memory, and system statistics.")
    if input("Do you want to use this source? (Y/n): ").strip().lower() not in ['y', '']:
        print("Skipping system entropy.")
        return b''

    # Gather entropy from system-related statistics
    try:
        load_avg = os.getloadavg()  # (1 min, 5 min, 15 min)
    except OSError:
        load_avg = (random.uniform(0, 10), random.uniform(0, 10), random.uniform(0, 10))  # Fallback for unsupported systems

    uptime = time.time() - os.stat('/proc/1').st_ctime if os.path.exists('/proc/1') else random.randint(0, 2**32)

    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()
        total_memory = int(next((line.split(":")[1].strip().split()[0] for line in meminfo.split("\n") if "MemTotal" in line), "0"))
        available_memory = int(next((line.split(":")[1].strip().split()[0] for line in meminfo.split("\n") if "MemAvailable" in line), "0"))
    except FileNotFoundError:
        total_memory = random.randint(2**30, 2**32)  # Random fallback
        available_memory = random.randint(0, total_memory)  # Random fallback

    # Count number of processes
    try:
        num_processes = len(os.listdir('/proc'))
    except FileNotFoundError:
        num_processes = random.randint(50, 500)  # Random fallback for non-Linux

    # Gather entropy from CPU usage (Linux only)
    try:
        with open("/proc/stat", "r") as f:
            cpu_stat = f.readline().split()
            cpu_usage = sum(map(int, cpu_stat[1:])) % 100  # Extracted CPU usage as a rough metric
    except FileNotFoundError:
        cpu_usage = random.randint(0, 100)  # Random fallback

    # Convert values to bytes
    data = (
        struct.pack("!Q", total_memory) +
        struct.pack("!Q", available_memory) +
        struct.pack("!Q", int(cpu_usage * 100)) +
        struct.pack("!Q", int(load_avg[0] * 100)) +
        struct.pack("!Q", int(load_avg[1] * 100)) +
        struct.pack("!Q", int(load_avg[2] * 100)) +
        struct.pack("!Q", int(uptime)) +
        struct.pack("!Q", num_processes)
    )

    # **Expand entropy size**
    while len(data) < 256:  # Extend to at least 256 bytes
        data += os.urandom(8) 

    # **Shuffle bytes randomly to break patterns**
    data = bytearray(data)
    random.shuffle(data)

    # Final SHA-512 hash to distribute randomness
    final_entropy = hashlib.shake_256(data).digest(256)
    hex_viewer(final_entropy)
    return final_entropy


def generate_entropy():
    entropy_sources = [
        #("      Keyboard", get_keyboard_entropy()),
        ("  Hardware RNG", get_hardware_rng()),
        ("   Yubikey RNG", get_yubikey_rng()),
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
    hex_viewer(combined_sources, False)

    print(f"\n\nShannon entropy calculations\n")
    for source_name, entropy in entropy_sources:
        entropy_shannon = shannon_entropy(entropy)
        print(f"{source_name}: {entropy_shannon}")


    print(f"\n\nShannon Entropy Scale Guide\n"
        f"7.98 - 8.00 → Near-perfect randomness (Nuclear launch codes)\n"
        f"7.80 - 7.98 → Excellent randomness (Cryptographic apps)\n"
        f"7.50 - 7.80 → Strong randomness (Security-sensitive use)\n"
        f"7.00 - 7.50 → Moderate randomness (Acceptable but could be improved)\n"
        f"Below  7.00 → Weak randomness (not suitable, patterns likely)")

    print(f"\n\nFINAL COMBINED SHANNON ENTROPY: {final_entropy_shannon}\n")

    input("\nPress ANY KEY to generate keys")  # Wait for user input before proceeding


    return combined_sources


SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141  # Order of secp256k1

def create_ethereum_keys(entropy):
    print("Validating Private Key Generation...")

    attempt = 0
    while True:
        attempt += 1

        # OLD METHOD - hashed_entropy = hashlib.sha3_256(entropy).digest()
        # **Instead of hashing once, mix entropy more effectively**
        
        mixed_entropy = hashlib.sha3_512(entropy).digest()  # SHA3-512 outputs 64 bytes
        private_key_bytes = mixed_entropy[:32]  # Use the first 32 bytes
        # The first 32 bytes are used, but the extra 32 bytes allow entropy reuse for retries - if needed.

        private_key_int = int.from_bytes(private_key_bytes, byteorder='big')

        if 1 <= private_key_int < SECP256K1_N:
            print(f"Private key is valid within secp256k1 range (Attempt {attempt})")
            break
        else:
            entropy = mixed_entropy  # Use mixed entropy and retry

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
    os.system("clear && printf '\x1b[3J'")
    sys.exit()


if __name__ == '__main__':
    entropy = generate_entropy()
    
    print("\nSeeding key generation")
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
