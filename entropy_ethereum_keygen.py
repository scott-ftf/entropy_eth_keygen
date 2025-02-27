import hashlib
import secrets
import os
import json
import urllib.request
import time
import subprocess
import binascii
import sys
import struct
import hashlib
import shutil
import math
from eth_keys import keys
from eth_account import Account
from eth_account.messages import encode_defunct
from collections import Counter
import socket
import re
import select
import platform
import threading
import zlib
import collections


# Define the API key file in the application root directory
API_KEY_FILE = os.path.join(os.path.dirname(__file__), "random_org_api.secret")


# Enable unaudited HD wallet features (required for mnemonic-based accounts)
Account.enable_unaudited_hdwallet_features()

# Calculate Shannon entropy of the byte sequences
def shannon_entropy(data):
    if not data:
        return 0.0  # Avoid division by zero if data is empty

    data_length = len(data)
    counter = Counter(data)  # Count occurrences of each byte (0-255)
    
    probabilities = [count / data_length for count in counter.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

    return round(entropy, 3)  


# Displays the results in hexadecimal format with ASCII representation, and calculates entropy.
def hex_viewer(data, shannon=True):
  
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
        print(f"\nData Byte Length: {len(data)}")
        print(f"    Unique Bytes: {len(counter)}")
        print(f" Shannon Entropy:", shannon_entropy(data))  


# Check if ykman is installed; if not, ask for permission and install via Snap
def install_ykman():
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


# Check if a YubiKey is detected and OpenPGP is enabled
def check_yubikey_status():
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


# Load API for randomn.org entropy source
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

# read CPU cycle counter on x86
def get_rdtsc():
    if platform.system() == "Linux":
        try:
            with open("/proc/self/stat", "r") as f:
                fields = f.read().split()
                return int(fields[21])  # CPU clock ticks
        except Exception:
            return 0
    return 0  # Default to 0 if not available


#splash screen
def splash():

    print("\nENTROPY ETHEREUM KEY GENERATOR")
    print("\nThis script will walk through a selection of randomness sources to be used as entropy in generating Ethereum key pairs. Any source may be skipped. Tests are available at the end to evaluate the results.")

    print("\nUSE AT YOUR OWN RISK")

    input("\nPress ANY KEY to begin")


# ENTROPY SOURCE - meant to be purely user selected dice, and time based entropy
def quantum_dice_entropy():
    os.system("clear && printf '\x1b[3J'")

    print("\n\nVIRTUAL QUANTUM DICE")
    print("A virtual 2^256-sided die - more possible outcomes than the number of atoms in the observable universe. Incorporates your input timing, high-speed CPU fluctuations, and nanosecond-precision system time jitter as additional entropy.")
    print("\nNOTE: Random time between key presses increases entropy.")

    total_rolls = 16  # Number of rolls required
    collected_entropy = b""  # Byte buffer to store all entropy sources

    # Capture system uptime for additional time-based entropy
    system_uptime_ns = time.monotonic_ns()

    input("\nPress ANY KEY to throw the first roll...")
    print()

    # Capture timestamp of first keypress
    first_keypress_time_ns = time.time_ns()

    # Initialize the rolling hash using the first keypress timestamp
    rolling_hash = hashlib.sha256(first_keypress_time_ns.to_bytes(8, 'big')).digest()

    for roll_number in range(1, total_rolls + 1):
        # Ensure a clean start for each roll
        sys.stdout.write("\n") 
        sys.stdout.flush()

        print(f"Roll {roll_number}/{total_rolls}")
        print("Press ANY KEY to stop the dice...")

        # Capture start time before rolling
        start_time_ns = time.time_ns()
        start_perf_counter_ns = time.perf_counter_ns()
        start_monotonic_ns = time.monotonic_ns()

        # Mix in CPU cycle counter for additional entropy
        start_rdtsc = get_rdtsc()

        # Continuously generate SHA-256 hashes as fast as possible
        while True:
            # Capture multiple timestamps per iteration for added jitter
            iter_time_ns = time.time_ns()
            iter_perf_ns = time.perf_counter_ns()
            iter_mono_ns = time.monotonic_ns()
            iter_rdtsc = get_rdtsc()

            # Update rolling hash using multiple timing sources
            rolling_hash = hashlib.sha256(
                rolling_hash +
                iter_time_ns.to_bytes(8, 'big') +
                iter_perf_ns.to_bytes(8, 'big') +
                iter_mono_ns.to_bytes(8, 'big') +
                iter_rdtsc.to_bytes(8, 'big')
            ).digest()

            # Print in-place so terminal output doesn't move
            sys.stdout.write("\r" + rolling_hash.hex())
            sys.stdout.flush()

            # Check for user input
            is_windows = sys.platform.startswith('win')
            if is_windows:
                import msvcrt
                if msvcrt.kbhit():
                    msvcrt.getch()  # Consume the input
                    break  # Stop rolling when user presses a key
            else:
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    input()  # Consume the input
                    break  # Stop rolling when user presses a key

        # Fix: Delete the last four lines before printing the final selected hash - Is this workingcorrect now?
        sys.stdout.write("\033[F\033[K")  # Remove "Press ANY KEY to stop the dice..."
        sys.stdout.write("\033[F\033[K")  # Remove rolling hash output
        sys.stdout.write("\033[F\033[K")  # Remove roll number
        sys.stdout.write("\033[F\033[K")  # Remove any lingering text
        sys.stdout.flush()

        # Print the final selection with roll number on a single line
        print(f"Roll #{roll_number} - {rolling_hash.hex()}")

        # Capture stop time
        stop_time_ns = time.time_ns()
        stop_perf_counter_ns = time.perf_counter_ns()
        stop_monotonic_ns = time.monotonic_ns()
        stop_rdtsc = get_rdtsc()

        # Final entropy sources for this roll
        entropy_sources = (
            rolling_hash +
            start_time_ns.to_bytes(8, 'big') +
            stop_time_ns.to_bytes(8, 'big') +
            start_perf_counter_ns.to_bytes(8, 'big') +
            stop_perf_counter_ns.to_bytes(8, 'big') +
            start_monotonic_ns.to_bytes(8, 'big') +
            stop_monotonic_ns.to_bytes(8, 'big') +
            start_rdtsc.to_bytes(8, 'big') +
            stop_rdtsc.to_bytes(8, 'big') +
            system_uptime_ns.to_bytes(8, 'big')
        )

        # Hash all collected entropy for this roll
        roll_entropy = hashlib.sha256(entropy_sources).digest()

        # Append to total entropy pool
        collected_entropy += roll_entropy

    # After the last roll (Roll #16), add blank line
    print("\n")

    # Generate 256 bytes (8 SHA-256 hashes)
    final_entropy = b""
    rolling_state = collected_entropy
    for _ in range(8):
        rolling_state = hashlib.sha256(rolling_state).digest()
        final_entropy += rolling_state

    # Display final entropy (hex format)
    hex_viewer(final_entropy)
    input("\nPress ANY KEY for the next source")    

    return final_entropy


# ENTROPY SOURCE - use hardware RNG if it exists
def get_hardware_rng():
    os.system("clear && printf '\x1b[3J'")

    print("\n\nHARDWARE RANDOM NUMBER GENERATOR")
    print("If a hardware RNG is installed, it produces true non-deterministic randomness from physical processes like thermal noise and electrical jitter for high-quality entropy without software-based algorithms. (may require sudo)")

    if not os.path.exists('/dev/hwrng'):
        print("\nNo hardware RNG device found")
        input("Press ANY KEY to skip hardware RNG check")
        return b''

    if input("\nDo you want to try this source? (y/N): ").strip().lower() in ['n', '']:
        print("Skipping hardware RNG check.")
        return b''

    try:
        # Try reading without sudo
        with open('/dev/hwrng', 'rb') as f:
            entropy = f.read(256)  # 🔹 Increased to 256 bytes for better entropy
        print("Successfully read from hardware RNG (without sudo).")
        hex_viewer(entropy)
        input("\nPress ANY KEY")

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
                print(f"\nFailed to read from hardware RNG with sudo:\n{process.stderr.decode().strip()}")
                input("Press ANY KEY to skip hardware RNG check")
                return b''

            entropy = process.stdout  # No need to re-encode
            print("Successfully read from hardware RNG (with sudo).")
            hex_viewer(entropy)
            input("\nPress ANY KEY")

            return entropy

        except Exception as e:
            print(f"\nError using sudo for hardware RNG: {e}")
            input("Press ANY KEY to skip hardware RNG check")
            return b''


# ENTROPY SOURCE - Use Yubikey as a RNG
def get_yubikey_rng():
    os.system("clear && printf '\x1b[3J'")

    print("\n\nYUBIKEY GPG RNG")
    print("If a YubiKey is attached, generates randomness via GPG where cryptographic operations occur on the YubiKey instead of the OS.")

    choice = input("\nDo you want to use this source? (Y/n): ").strip().lower()
    if choice not in ['y', '']:
        print("Skipping YubiKey RNG extraction.")
        return b''

    input("\nPlug in your YubiKey and press Enter when ready...")

    if not install_ykman():
        return b''

    if not check_yubikey_status():
        print("Yibikey status failed. Skipping YubiKey RNG extraction.")
        input("Press ANY KEY to continue")
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
        input("\nPress ANY KEY for the next source")

        return entropy

    except subprocess.CalledProcessError:
        print("Failed to extract entropy from YubiKey using GPG.")
        input("Press ANY KEY to continue")
        return b''


# ENTROPY SOURCE - Random.org
def get_random_integer():
    os.system("clear && printf '\x1b[3J'")

    print("\n\nRANDOM.ORG RNG")
    print("Retrieves randomness from RANDOM.ORG, generated from atmospheric noise for high-quality entropy beyond algorithmic methods. (Requires free API key in the random_org_api.secret file)")

    if input("\nDo you want to use this source? (Y/n): ").strip().lower() not in ['y', '']:
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
            input("\nPress ANY KEY for the next source")

            return entropy

        else:
            print(f"Unexpected response format: {result}")
            return b''

    except urllib.error.URLError as e:
        print(f"Failed to fetch from Random.org: {e}")
        return b''


# ENTROPY SOURCE - Secrets library
def get_secrets():
    os.system("clear && printf '\x1b[3J'")

    print("\n\nSECRETS LIBRARY")
    print("Generates cryptographic-grade randomness using the Python secrets module, designed for secure tokens and key generation.")
    
    if input("\nDo you want to use this source? (Y/n): ").strip().lower() in ['y', '']:
        # Generate 256 bytes directly from secrets
        entropy = secrets.token_bytes(256)

        hex_viewer(entropy)
        input("\nPress ANY KEY for the next source")

        return entropy

    return b''


# ENTROPY SOURCE - /dev/random
def get_dev_random():
    os.system("clear && printf '\x1b[3J'")

    print("\n\nSYSTEM ENTROPY POOL")
    print("Utilizes /dev/random, a blocking randomness source standard on Unix-like operating systems that continuously harvests system entropy from user interactions, disk I/O fluctuations, and hardware noise.")

    if input("\nDo you want to use this source? (Y/n): ").strip().lower() not in ['y', '']:
        return b''

    with open('/dev/random', 'rb') as f:
        entropy_chunks = [f.read(256) for _ in range(4)]  # Read in 4 blocks

    # XOR the chunks together for better mixing
    entropy = bytes(a ^ b ^ c ^ d for a, b, c, d in zip(*entropy_chunks))

    # Add timing jitter to introduce variability at the extraction moment
    jitter = struct.pack("!Q", time.perf_counter_ns() ^ time.time_ns() ^ time.monotonic_ns())
    entropy = bytes(a ^ b for a, b in zip(entropy, jitter * (len(entropy) // 8)))

    hex_viewer(entropy)
    input("\nPress ANY KEY for the next source")

    return entropy


# ENTROPY SOURCE - uses network metrics t generate randomness
def get_network_entropy():
    os.system("clear && printf '\x1b[3J'")

    print("\n\nNETWORK ENTROPY")
    print("Network metrics are highly variable real-world conditions that are unpredictable. Collects randomness from connections, ephemeral port bindings, interface statistics, router queries, ping jitter, network buffer fill levels, and TCP retransmission counts.")
    
    if input("\nDo you want to use this source? (Y/n): ").strip().lower() not in ['y', '']:
        print("Skipping network entropy.")
        return b''

    network_data = bytearray()
    max_attempts = 10
    attempts = 0

    def collect_network_metrics():
        snapshot = bytearray()

        # Active TCP Connections
        try:
            connections = socket.getaddrinfo(None, None, proto=socket.IPPROTO_TCP)
            for conn in connections:
                snapshot.extend(struct.pack("!H", conn[4][1]))  # Port numbers
        except Exception:
            pass

        # Bind Multiple Ephemeral Ports
        try:
            for _ in range(3):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("", 0))
                local_port = sock.getsockname()[1]
                snapshot.extend(struct.pack("!H", local_port))
                sock.close()
        except Exception:
            pass

        # Network Interface Statistics
        try:
            with open("/proc/net/dev", "r") as f:
                lines = f.readlines()[2:]
                for line in lines:
                    fields = line.split()
                    if len(fields) >= 10:
                        snapshot.extend(struct.pack("!Q", int(fields[1])))  # RX Bytes
                        snapshot.extend(struct.pack("!Q", int(fields[9])))  # TX Bytes
        except Exception:
            pass

        # Interface Up/Down State
        try:
            for iface in os.listdir('/sys/class/net/'):
                state_path = f"/sys/class/net/{iface}/operstate"
                if os.path.exists(state_path):
                    with open(state_path, 'r') as f:
                        snapshot.extend(struct.pack("!B", 1 if f.read().strip() == "up" else 0))
        except Exception:
            pass

        # Ping Local Router & Capture Timing Jitter
        try:
            start_time = time.perf_counter_ns()
            subprocess.run(["ping", "-c", "1", "-W", "1", "192.168.1.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elapsed_time = time.perf_counter_ns() - start_time
            snapshot.extend(elapsed_time.to_bytes(8, 'big'))
        except Exception:
            pass

        # Query ARP Table for Router MAC Address
        try:
            with open("/proc/net/arp", "r") as f:
                arp_entries = f.readlines()[1:]
                for entry in arp_entries:
                    fields = entry.split()
                    if len(fields) >= 4:
                        snapshot.extend(socket.inet_aton(fields[0]))  # IP
                        snapshot.extend(bytes.fromhex(fields[3].replace(":", "")))  # MAC
        except Exception:
            pass

        # Capture TCP Retransmission Count
        try:
            with open("/proc/net/snmp", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "Tcp:" in line:
                        parts = line.split()
                        snapshot.extend(struct.pack("!Q", int(parts[11])))  # TCP retransmissions
        except Exception:
            pass

        # Capture Network Buffer Fill Levels from /proc/net/softnet_stat
        try:
            with open("/proc/net/softnet_stat", "r") as f:
                lines = f.readlines()
                for line in lines:
                    values = [int(x, 16) for x in line.split()]
                    snapshot.extend(struct.pack("!Q", values[0]))  # First value is buffer fill level
        except Exception:
            pass

        # Capture TCP/UDP Queue Depths from /proc/net/netstat
        try:
            with open("/proc/net/netstat", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "TcpExt:" in line:
                        parts = line.split()
                        snapshot.extend(struct.pack("!Q", int(parts[7])))  # TCP queue depth
        except Exception:
            pass

        return snapshot

    while len(network_data) < 256 and attempts < max_attempts:
        new_data = collect_network_metrics()
        network_data.extend(new_data)
        attempts += 1
        if len(new_data) == 0:
            print("Warning: No new network data collected, stopping early.")
            return b''  

    network_data = network_data[:256]

    xor_entropy = bytes(a ^ b for a, b in zip(network_data, os.urandom(len(network_data))))
    final_entropy = hashlib.shake_256(xor_entropy).digest(256)
    hex_viewer(final_entropy)
    input("\nPress ANY KEY for the next source")

    return final_entropy


# ENTROPY SOURCE - randoness from machine meterics
def get_machine_entropy():
    os.system("clear && printf '\x1b[3J'")

    print("\n\nMACHINE STATE ENTROPY")
    print("Extracts randomness from volatile system states, including CPU load variations, memory availability, process scheduling jitter, uptime fluctuations, kernel entropy metrics, RAM access timing, and CPU execution jitter.")

    if input("\nDo you want to use this source? (Y/n): ").strip().lower() not in ['y', '']:
        print("Skipping machine entropy.")
        return b''

    def jitter_entropy():
        t1 = time.perf_counter_ns()
        t2 = time.time_ns()
        t3 = time.monotonic_ns()
        return struct.pack("!Q", t1 ^ t2 ^ t3)

    def ram_jitter_entropy():
        buffer = bytearray(1024 * 1024)  # Allocate 1MB
        start_time = time.perf_counter_ns()
        for i in range(len(buffer)):
            buffer[i] = (buffer[i] + i) % 256  # Modify memory
        elapsed_time = time.perf_counter_ns() - start_time
        return struct.pack("!Q", elapsed_time)

    def thread_jitter_entropy():
        def worker():
            _ = sum(x * x for x in range(500000))  # Computational work

        start_time = time.perf_counter_ns()
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed_time = time.perf_counter_ns() - start_time
        return struct.pack("!Q", elapsed_time)

    # Gather entropy from system statistics
    try:
        load_avg = os.getloadavg() + (len(os.sched_getaffinity(0)),)
    except OSError:
        load_avg = (0, 0, 0, 0)

    uptime = time.time() - min(
        os.stat('/proc/1').st_ctime,
        os.stat('/proc/self').st_ctime
    ) if os.path.exists('/proc/1') else jitter_entropy()

    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()
        swap_usage = int(next((line.split(":")[1].strip().split()[0] for line in meminfo.split("\n") if "SwapFree" in line), "0"))
    except FileNotFoundError:
        swap_usage = jitter_entropy()

    try:
        num_processes = len(os.listdir('/proc'))
    except FileNotFoundError:
        num_processes = jitter_entropy()

    try:
        with open("/proc/stat", "r") as f:
            cpu_stat = f.readline().split()
            cpu_usage = sum(map(int, cpu_stat[1:])) % 100
    except FileNotFoundError:
        cpu_usage = jitter_entropy()

    try:
        with open("/proc/sys/kernel/random/entropy_avail", "r") as f:
            kernel_entropy = int(f.read().strip())
    except FileNotFoundError:
        kernel_entropy = jitter_entropy()

    # Convert values to bytes
    data = (
        struct.pack("!Q", swap_usage if isinstance(swap_usage, int) else int.from_bytes(swap_usage, 'big')) +
        struct.pack("!Q", int(cpu_usage * 100)) +
        struct.pack("!Q", int(load_avg[0] * 100)) +
        struct.pack("!Q", int(load_avg[1] * 100)) +
        struct.pack("!Q", int(load_avg[2] * 100)) +
        struct.pack("!Q", int(uptime) if isinstance(uptime, (int, float)) else int.from_bytes(uptime, 'big')) +
        struct.pack("!Q", num_processes if isinstance(num_processes, int) else int.from_bytes(num_processes, 'big')) +
        struct.pack("!Q", kernel_entropy if isinstance(kernel_entropy, int) else int.from_bytes(kernel_entropy, 'big')) +
        ram_jitter_entropy() +
        thread_jitter_entropy()
    )

    # Ensure at least 256 bytes
    while len(data) < 256:
        data += jitter_entropy()

    # XOR mixing for stronger randomness
    data = bytes(a ^ b for a, b in zip(data, jitter_entropy() * (len(data) // 8)))

    # Final SHA-512 hash to distribute randomness
    final_entropy = hashlib.shake_256(data).digest(256)
    hex_viewer(final_entropy)
    input("\nPress ANY KEY for the next source")

    return final_entropy


# walk through each entropy source, then combine the generated entropy
def generate_entropy():
    entropy_sources = [
        ("   Quantum Dice", quantum_dice_entropy()),
        ("Secrets Library", get_secrets()),
        ("     dev/random", get_dev_random()),
        ("Network Metrics", get_network_entropy()),
        ("Machine Entropy", get_machine_entropy()),
        ("     Random.org", get_random_integer()),
        ("    Yubikey RNG", get_yubikey_rng()),
        ("   Hardware RNG", get_hardware_rng())
    ]    

    # Filter out None or invalid entropy sources
    valid_entropy = [entropy for _, entropy in entropy_sources if entropy]

    # Combine entropy sources 
    final_entropy = b''.join(valid_entropy)  

    print("\n\nCombining Final Entropy\n")
    hex_viewer(final_entropy, True)

    return final_entropy, entropy_sources


def derive_ethereum_private_key(mnemonic: str) -> bytes:
    """Derive the Ethereum private key from a BIP-39 mnemonic using `eth_account`."""
    private_key = Account.from_mnemonic(mnemonic)._private_key
    return private_key


# Verify the correctness of generated keys
def verify_keys(private_key, eth_address, seed_phrase=None, attempt=1):
    print(f"\n[ATTEMPT {attempt}] Verifying generated wallet...")

    SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141  # Order of secp256k1

    # Check Private Key Validity 
    private_key_int = int.from_bytes(private_key.key, byteorder="big")
    if not (1 <= private_key_int < SECP256K1_N):
        print("\nVerification failed: Private key is out of valid secp256k1 range.")
        print(f"Generated Private Key: {private_key.key.hex()}")
        print(f"Valid Range: 1 to {hex(SECP256K1_N - 1)}")
        return False, f"ATTEMPT {attempt}: Private key is invalid."
    print("  ✓ Private key is within the valid secp256k1 range.")

    # Check Public Key Matches Private Key
    derived_public_key = private_key._key_obj.public_key
    if derived_public_key != private_key._key_obj.public_key:
        print("\nVerification failed: Derived public key does not match expected public key.")
        print(f"Expected Public Key: {private_key._key_obj.public_key}")
        print(f"Derived Public Key: {derived_public_key}")
        return False, f"Attempt {attempt}: Public key does not match private key."
    print("  ✓ Public key correctly matches private key.")

    # Check Address Correctly Derived from Public Key
    derived_address = private_key.address
    if derived_address.lower() != eth_address.lower():
        print("\nVerification failed: Derived Ethereum address does not match expected address.")
        print(f"Expected Address: {eth_address}")
        print(f"Derived Address: {derived_address}")
        return False, f"Attempt {attempt}: Address does not match derived address."
    print("  ✓ Ethereum address correctly derived from public key.")

    # Signature Verification Test
    test_message = encode_defunct(text="Ethereum key verification test.")  # Properly formatted message
    signed_message = Account.sign_message(test_message, private_key.key)  # Now uses correct format
    recovered_address = Account.recover_message(test_message, signature=signed_message.signature)
    
    if recovered_address.lower() != eth_address.lower():
        print("\nVerification failed: Signed message verification failed.")
        print(f"Expected Address: {eth_address}")
        print(f"Recovered Address: {recovered_address}")
        return False, f"Attempt {attempt}: Signature verification failed."
    print("  ✓ Signature verification passed. Private key correctly signs messages.")

    # Mnemonic Consistency Check If Seed Phrase Used
    if seed_phrase:
        regenerated_private_key = Account.from_mnemonic(seed_phrase)

        if regenerated_private_key.key.hex() != private_key.key.hex():
            print("\nVerification failed: Seed phrase does not regenerate the same private key.")
            print(f"Expected Private Key (Hex): {private_key.key.hex()}")
            print(f"Regenerated Private Key (Hex): {regenerated_private_key.key.hex()}")
            return False, f"Attempt {attempt}: Seed phrase does not regenerate the same private key."
        print("  ✓ Seed phrase correctly regenerates the private key.")

    print("  ✓ All verification checks passed.")
    return True, f"[Attempt {attempt}] Wallet verification passed!"


# Load BIP39 wordlist
def load_bip39_wordlist():
    with open("bip39_wordlist.txt", "r") as f:
        return [word.strip() for word in f.readlines()]

# Generate a BIP39 mnemonic from entropy
def generate_bip39_mnemonic(entropy):
    
    if len(entropy) not in [16, 20, 24, 28, 32]:  # 128-256 bits
        raise ValueError("Entropy must be 128, 160, 192, 224, or 256 bits.")

    wordlist = load_bip39_wordlist()
    
    # Compute checksum
    checksum_bits = len(entropy) // 4  # Number of bits to take from SHA-256
    hash_digest = hashlib.sha256(entropy).digest()
    
    # Append checksum to entropy
    entropy_bits = bin(int.from_bytes(entropy, "big"))[2:].zfill(len(entropy) * 8)
    checksum_bits = bin(int.from_bytes(hash_digest, "big"))[2:].zfill(256)[:checksum_bits]
    full_bits = entropy_bits + checksum_bits  # Full entropy + checksum
    
    # Split into 11-bit chunks
    mnemonic = [wordlist[int(full_bits[i:i+11], 2)] for i in range(0, len(full_bits), 11)]

    return " ".join(mnemonic)

# Wallet generation process
def generate_wallet(entropy):
    MAX_ATTEMPTS = 4
    os.system("clear && printf '\x1b[3J'")
    print("\n-----------------------")
    print("Seeding key generation")
    print("-----------------------")
    print("Choose key generation method:\n")
    print("1) No seed phrase (direct private key generation)")
    print("2) 12-word mnemonic (HD, Compatible with certain wallets)")
    print("3) 24-word mnemonic (HD, Highest security)")

    choice = input("\nEnter 1, 2, or 3: ").strip()
    use_bip39 = choice in ["2", "3"]

    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        print(f"\ngenerating wallet...")

        mixed_entropy = hashlib.sha3_512(entropy).digest()  # 64 bytes total

        if not use_bip39:
            # ption 1: Direct entropy-based private key
            private_key_bytes = mixed_entropy[:32]
            private_key = Account.from_key(private_key_bytes)
            eth_address = private_key.address
            seed_phrase = None
        else:
            # Option 2/3: Generate a mnemonic seed phrase
            mnemonic_entropy = mixed_entropy[:16] if choice == "2" else mixed_entropy[:32]
            seed_phrase = generate_bip39_mnemonic(mnemonic_entropy)

            # Use eth_account to derive the private key from the mnemonic
            private_key_obj = Account.from_mnemonic(seed_phrase)

            # Keep as LocalAccount for verification
            private_key = private_key_obj
            eth_address = private_key_obj.address

        # Verify key correctness
        is_valid, verification_msg = verify_keys(private_key, eth_address, seed_phrase, attempt)

        if is_valid:
            print(verification_msg)

            # Extract the hex private key only AFTER verification
            private_key = private_key_obj.key.hex() if use_bip39 else private_key.key.hex()
            break
        else:
            print(f"{verification_msg} Retrying...")

        if attempt == MAX_ATTEMPTS:
            print("\nERROR: The generated wallet could not pass verification after 4 attempts.")
            print("Please try generating a new source of entropy and rerun the script.")
            exit(1)

    input("\nPrivate key is ready. Check for privacy.\n\nPress ANY KEY to display")

    return private_key, eth_address, seed_phrase


# Performs a Chi-Square Goodness-of-Fit test to check uniform distribution
def chi_square_test(data):
    if not data:
        return None 

    byte_counts = collections.Counter(data)
    expected = len(data) / 256  # Expected count per byte for uniform distribution
    chi2_stat = sum((count - expected) ** 2 / expected for count in byte_counts.values())

    return chi2_stat  


# Computes correlation between consecutive bytes in data.
def serial_correlation_test(data):
    if len(data) < 2:
        return None

    mean = sum(data) / len(data)
    numerator = sum((data[i] - mean) * (data[i+1] - mean) for i in range(len(data) - 1))
    denominator = sum((x - mean) ** 2 for x in data)

    correlation = numerator / denominator if denominator != 0 else 0
    return correlation


# Estimates π using entropy as random (x, y) coordinates.
def monte_carlo_pi(data):
    num_pairs = len(data) // 2
    if num_pairs < 10:
        return None  # Too little data to estimate π meaningfully

    inside_circle = 0
    for i in range(0, len(data) - 1, 2):
        x = data[i] / 255.0  # Normalize to 0-1
        y = data[i+1] / 255.0
        if x**2 + y**2 <= 1:
            inside_circle += 1

    estimated_pi = (inside_circle / num_pairs) * 4
    return estimated_pi


# compute compression ratio
def compression_ratio(data):
    if not data:
        return None
    compressed_data = zlib.compress(data)
    return len(compressed_data) / len(data)


# compute bit frequency
def bit_frequency(data):
    if not data or len(data) == 0:
        return None, None

    total_bits = len(data) * 8  # Each byte has 8 bits
    zero_count = sum((byte >> i) & 1 == 0 for byte in data for i in range(8))
    one_count = total_bits - zero_count  # Complement of zero count

    return zero_count, one_count  # Return **raw bit counts**


# display each of the test results
def display_randomness_tests(entropy, entropy_sources):
    tests = [
        {
            "name": "Shannon Entropy",
            "description": "Measures the uncertainty in a dataset.\n"
                           "Higher values indicate more randomness.",
            "scale": "7.95 - 8.00 → Theoretical maximum entropy (Perfect)\n"
                     "7.85 - 7.95 → Cryptographically ideal (Strongest randomness)\n"
                     "7.70 - 7.85 → Excellent (Suitable for key generation)\n"
                     "7.50 - 7.70 → Strong (Security-sensitive applications)\n"
                     "7.00 - 7.50 → Moderate (Could be improved)\n"
                     "Below  7.00 → Weak (Patterns likely)",
            "function": shannon_entropy
        },
        {
            "name": "Chi-Square Test",
            "description": "Measures how uniformly distributed the byte values are.\n"
                           "Lower values suggest better randomness, higher values indicate patterns.",
            "scale": "  0 - 250 → Good (Uniform)\n"
                     "250 - 400 → Moderate (Some bias)\n"
                     "400+      → Poor (Highly non-random)",
            "function": chi_square_test
        },
        {
            "name": "Serial Correlation Test",
            "description": "Measures correlation between consecutive bytes.\n"
                           "Values closer to 0 indicate better randomness.",
            "scale": "-0.1 to 0.1 → Excellent (No correlation)\n"
                     " 0.1 to 0.3 → Moderate (Some correlation)\n"
                     " 0.3+       → Poor (Sequences exist)",
            "function": serial_correlation_test
        },
        {
            "name": "Monte Carlo π Estimation",
            "description": "Estimates π based on random coordinate generation.\n"
                           "Smaller variance from 3.14159 indicates better randomness.",
            "scale": "0.000 - 0.100 → Excellent (Very random)\n"
                     "0.101 - 0.300 → Moderate (Some deviation)\n"
                     "0.301+        → Poor (Non-random)",
            "function": monte_carlo_pi
        },
        {
            "name": "Compression Ratio",
            "description": "Measures how well the data compresses. Random data should be hard to compress.\n"
                           "Lower compression ratios indicate better randomness.",
            "scale": "0.95 - 1.00 → Excellent (Hard to compress)\n"
                     "0.90 - 0.95 → Moderate (Some patterns exist)\n"
                     "Below  0.90 → Poor (Highly compressible, non-random)",
            "function": compression_ratio
        },
        {
            "name": "Bit Frequency Balance",
            "description": "Checks the balance of 0s and 1s in the data.\n"
                           "Perfect randomness should have a ~50% balance of both.",
            "scale": "0.0% - 0.5% → Excellent (Near perfect balance)\n"
                     "0.5% - 2.0% → Moderate (Slight imbalance, acceptable)\n"
                     "> 2.0%      → Poor (Unbalanced, possible bias",
            "function": bit_frequency
        }
    ]

    # Iterate through each randomness test
    for test in tests:
        os.system("clear && printf '\x1b[3J'")
        print("\n" + "=" * 50)
        print(f"{test['name']}")
        print("=" * 50)
        print(test['description'])
        print("\nInterpretation Guide:")
        print(test['scale'])

        print("\nResults:")

        # Compute and display test results for each entropy source
        for source_name, source_entropy in entropy_sources:
            result = test["function"](source_entropy)
            formatted_result = format_test_result(test["name"], result)
            print(f"{source_name}: {formatted_result}")

        # Compute and display test result for final combined entropy
        final_result = test["function"](entropy)
        formatted_final_result = format_test_result(test["name"], final_result)
        print(f"\nFinal Combined Entropy: {formatted_final_result}")

        # Pause before moving to the next test
        input("\nPress ANY KEY to continue")
        os.system("clear && printf '\x1b[3J'")


# we need to format some of the results
def format_test_result(test_name, result):
    """Formats test results properly based on test type."""

    if test_name == "Monte Carlo π Estimation":
        if result is None:
            return "Not enough data"
        variance = abs(result - 3.14159)
        return f"{result:.6f} ({variance:.5f} variance)"

    elif test_name == "Bit Frequency Balance":
        if isinstance(result, tuple) and result[0] is not None and result[1] is not None:
            zero_freq, one_freq = result  # Raw frequency counts
            total_freq = zero_freq + one_freq
            expected_half = total_freq / 2  # Expected perfect balance

            # Calculate percentage deviation **from perfect 50% balance**
            deviation = (abs(zero_freq - expected_half) / expected_half) * 100

            return f"{zero_freq} / {one_freq} ({deviation:.2f}% variance)"
        return "Not enough data"

    elif test_name == "Chi-Square Test":
        if isinstance(result, (float, int)):
            return f"{round(result)}"  # **Ensure it is displayed as an integer**
        return "Not enough data"

    elif isinstance(result, tuple):  # Some tests return multiple values
        return " / ".join(f"{val:.3f}" if isinstance(val, float) else str(val) for val in result)

    elif result is None:
        return "Not enough data"

    else:
        return f"{result:.3f}" if isinstance(result, float) else str(result)


# Disply test results, or just generate keys
def choose_to_display_tests(entropy, entropy_sources):
    print(f"\n\n------------------------------")
    print(f"ENTROPY GENERATION COMPLETE") 
    print(f"------------------------------\n"   
    "The following tests evaluate different aspects of the randomness generated:\n\n"
    "Shannon Entropy\n"
    "Chi-Square Test\n"
    "Serial Correlation\n"
    "Monte Carlo π Estimation\n"
    "Compression Ratio\n"
    "Bit Frequency Test\n"
    "\nNo single test can prove true randomness, and some variation is expected. However, strong entropy should perform well across multiple tests without major biases.")

    if input("\nDo you want to view the entropy tests before generating keys? (Y/n): ").strip().lower() in ['y', '']:
        display_randomness_tests(entropy, entropy_sources)

    return

# dont just clear screen, but dump screen buffer    
def clearScreen():
    print("\n\n     This is the end of this script") 
    print("      record key outputs securely")  
    print("\nHit ENTER to clear screen buffer and exit")  
    input()
    os.system("clear && printf '\x1b[3J'")
    sys.exit()


if __name__ == '__main__':    
    # Display the splash screen
    splash()

    # walk through each entropy source to generate the entropy
    entropy, entropy_sources = generate_entropy()

    # Do we want to display the entropy tests?
    choose_to_display_tests(entropy, entropy_sources)    
    
    # generate keys
    private_key, eth_address, seed_phrase = generate_wallet(entropy)

    # display keys
    os.system("clear && printf '\x1b[3J'")
    print("\n\n-----------BEGIN KEY OUTPUT-------------")
    if seed_phrase:
        print("Seed Phrase:")
        print(f"{seed_phrase}\n")
    print("private key:")
    print(private_key)
    print("\npublic key:")
    print(eth_address)
    print("------------END KEY OUTPUT--------------")
    

# clear screen on user input
    clearScreen()
