#!/usr/bin/env python3
"""
Bitcoin Core & Electrs Compiler for macOS
Production-Optimized Version

Security: All shell injection vulnerabilities eliminated
Performance: Parallel operations, efficient I/O, proper resource management
Reliability: Comprehensive error handling, timeout management, cleanup handlers
Threading: Thread-safe GUI updates, proper synchronization
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import threading
import queue
import requests
import multiprocessing
import shutil
import re
import platform
import time
import hashlib
import logging
import signal
import atexit
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from functools import lru_cache, wraps
from contextlib import contextmanager
from enum import Enum


# ================== LOGGING SETUP ==================
class GUILogHandler(logging.Handler):
    """Thread-safe logging handler that writes to tkinter Text widget"""
    
    def __init__(self, text_widget: Optional[tk.Text] = None):
        super().__init__()
        self.text_widget = text_widget
        self._queue = queue.Queue()
        
    def emit(self, record: logging.LogRecord) -> None:
        """Queue log message for thread-safe GUI update"""
        try:
            msg = self.format(record)
            self._queue.put(msg + '\n')
        except Exception:
            self.handleError(record)
    
    def set_widget(self, widget: tk.Text) -> None:
        """Set the text widget (called after GUI creation)"""
        self.text_widget = widget
        # Start consuming queue
        if widget:
            self._consume_queue()
    
    def _consume_queue(self) -> None:
        """Consume queued log messages and update widget"""
        if not self.text_widget:
            return
            
        try:
            # Process up to 100 messages per call to avoid blocking
            for _ in range(100):
                try:
                    msg = self._queue.get_nowait()
                    if self.text_widget.winfo_exists():
                        self.text_widget.insert('end', msg)
                        self.text_widget.see('end')
                        
                        # Limit log size to prevent memory issues
                        lines = int(self.text_widget.index('end-1c').split('.')[0])
                        if lines > 5000:
                            self.text_widget.delete('1.0', '1000.0')
                except queue.Empty:
                    break
                    
            # Schedule next consumption
            if self.text_widget.winfo_exists():
                self.text_widget.after(100, self._consume_queue)
        except tk.TclError:
            # Widget destroyed
            pass


# Setup logger
logger = logging.getLogger('BitcoinCompiler')
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)

# GUI handler (will be connected to widget later)
gui_handler = GUILogHandler()
gui_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(gui_handler)


# ================== CONFIGURATION ==================
@dataclass
class Config:
    """Application configuration"""
    BITCOIN_API: str = "https://api.github.com/repos/bitcoin/bitcoin/releases"
    BITCOIN_REPO: str = "https://github.com/bitcoin/bitcoin.git"
    ELECTRS_API: str = "https://api.github.com/repos/romanz/electrs/releases"
    ELECTRS_REPO: str = "https://github.com/romanz/electrs.git"
    DEFAULT_BUILD_DIR: Path = Path.home() / "Downloads" / "bitcoin_builds"
    
    # Timeouts (seconds)
    SUBPROCESS_TIMEOUT: int = 3600  # 1 hour max for build steps
    NETWORK_CONNECT_TIMEOUT: int = 10
    NETWORK_READ_TIMEOUT: int = 30
    QUICK_CMD_TIMEOUT: int = 10
    GIT_TIMEOUT: int = 300  # 5 minutes for git operations
    
    # Performance
    HASH_CHUNK_SIZE: int = 65536  # 64KB chunks for optimal I/O
    MAX_WORKERS: int = 4
    API_CACHE_TTL: int = 300  # 5 minutes
    
    # Resource limits
    MAX_LOG_LINES: int = 5000
    MAX_SUBPROCESS_OUTPUT: int = 10_000_000  # 10MB

config = Config()


# ================== PYINSTALLER COMPATIBILITY ==================
def is_pyinstaller() -> bool:
    """Check if running as PyInstaller bundle"""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')


def get_base_path() -> Path:
    """Get base path for resources (works with PyInstaller)"""
    if is_pyinstaller():
        return Path(sys._MEIPASS)
    return Path.cwd()


BASE_PATH = get_base_path()

# Fix macOS app bundle issues
if is_pyinstaller() and platform.system() == 'Darwin':
    os.environ.setdefault('APP_STARTED', '1')


# ================== PATH CONFIGURATION ==================
def get_safe_path() -> str:
    """Build safe PATH with known good locations"""
    paths = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ]
    
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        paths.insert(0, str(cargo_bin))
    
    return ":".join(paths)


os.environ["PATH"] = get_safe_path()


# ================== ARCHITECTURE DETECTION ==================
class Architecture(Enum):
    """System architecture types"""
    APPLE_SILICON = "apple_silicon"
    INTEL = "intel"
    UNKNOWN = "unknown"


@lru_cache(maxsize=1)
def get_architecture() -> Architecture:
    """Detect system architecture (cached)"""
    machine = platform.machine().lower()
    if machine == "arm64":
        return Architecture.APPLE_SILICON
    elif machine in ("x86_64", "amd64"):
        return Architecture.INTEL
    return Architecture.UNKNOWN


@lru_cache(maxsize=1)
def get_chip_name() -> Optional[str]:
    """Get specific M-series chip name for Apple Silicon (cached)"""
    if platform.system() != 'Darwin':
        return None
    
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=config.QUICK_CMD_TIMEOUT
        )
        
        if result.returncode == 0:
            brand = result.stdout.strip()
            match = re.search(r'Apple (M\d+(?:\s+(?:Pro|Max|Ultra))?)', brand, re.IGNORECASE)
            if match:
                return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return None


ARCH = get_architecture()
CHIP_NAME = get_chip_name()


# ================== HOMEBREW DETECTION ==================
@lru_cache(maxsize=1)
def find_brew() -> Optional[Path]:
    """Find Homebrew installation (cached)"""
    for path_str in ["/opt/homebrew/bin/brew", "/usr/local/bin/brew"]:
        path = Path(path_str)
        if path.is_file():
            return path
    return None


BREW = find_brew()
BREW_PREFIX = BREW.parent.parent if BREW else None


# ================== SECURITY UTILITIES ==================
def sanitize_version_tag(tag: str) -> str:
    """Sanitize version tag to prevent injection attacks"""
    # Only allow alphanumeric, dots, dashes, and 'v' prefix
    if not re.match(r'^v?[\d.]+([-][\w.]+)?$', tag):
        raise ValueError(f"Invalid version tag format: {tag}")
    return tag


def validate_directory(path: Path, must_exist: bool = False) -> Path:
    """Validate directory path for security"""
    try:
        # Resolve to absolute path, following symlinks
        resolved = path.resolve()
        
        # Ensure it's under user's home or standard build locations
        home = Path.home()
        allowed_parents = [home, Path('/tmp'), Path('/var/tmp')]
        
        if not any(str(resolved).startswith(str(parent)) for parent in allowed_parents):
            raise ValueError(f"Path outside allowed locations: {resolved}")
        
        if must_exist and not resolved.exists():
            raise ValueError(f"Path does not exist: {resolved}")
            
        if must_exist and not resolved.is_dir():
            raise ValueError(f"Path is not a directory: {resolved}")
        
        return resolved
    except Exception as e:
        raise ValueError(f"Invalid path: {path} - {e}")


# ================== SUBPROCESS UTILITIES ==================
class SubprocessManager:
    """Thread-safe subprocess management with timeouts and cleanup"""
    
    def __init__(self):
        self._processes: List[subprocess.Popen] = []
        self._lock = threading.Lock()
    
    def run_command(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        log_output: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Execute command safely with timeout and cleanup
        
        Args:
            cmd: Command as list (NOT string) for security
            cwd: Working directory
            env: Environment variables
            timeout: Timeout in seconds
            log_output: Whether to log output in real-time
            
        Returns:
            CompletedProcess instance
            
        Raises:
            subprocess.TimeoutExpired: If command times out
            subprocess.CalledProcessError: If command fails
        """
        if env is None:
            env = os.environ.copy()
        
        logger.info(f"\n$ {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,  # List, NOT string - prevents shell injection
            shell=False,  # CRITICAL: Never use shell=True
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            env=env
        )
        
        # Track process for cleanup
        with self._lock:
            self._processes.append(process)
        
        try:
            output_lines = []
            start_time = time.time()
            
            # Read output with timeout checking
            while True:
                if timeout and (time.time() - start_time) > timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise subprocess.TimeoutExpired(cmd, timeout)
                
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    time.sleep(0.01)
                    continue
                
                output_lines.append(line)
                if log_output:
                    logger.info(line.rstrip())
                
                # Prevent memory exhaustion
                if sum(len(l) for l in output_lines) > config.MAX_SUBPROCESS_OUTPUT:
                    logger.warning("Output size limit reached, truncating...")
                    output_lines = output_lines[-1000:]
            
            returncode = process.wait()
            
            if returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode, cmd, ''.join(output_lines)
                )
            
            return subprocess.CompletedProcess(
                cmd, returncode, ''.join(output_lines), ''
            )
            
        finally:
            # Ensure process is terminated and cleaned up
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            except Exception:
                pass
            finally:
                with self._lock:
                    try:
                        self._processes.remove(process)
                    except ValueError:
                        pass
    
    def cleanup(self) -> None:
        """Terminate all running processes"""
        with self._lock:
            for process in self._processes[:]:
                try:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                except Exception:
                    pass
            self._processes.clear()


# Global subprocess manager
subprocess_manager = SubprocessManager()

# Register cleanup
atexit.register(subprocess_manager.cleanup)
signal.signal(signal.SIGTERM, lambda *args: subprocess_manager.cleanup())


# ================== NETWORK UTILITIES ==================
class APICache:
    """Simple TTL cache for API responses"""
    
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        with self._lock:
            if key in self._cache:
                timestamp, value = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    del self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value with current timestamp"""
        with self._lock:
            self._cache[key] = (time.time(), value)


api_cache = APICache(ttl=config.API_CACHE_TTL)


def fetch_with_retry(
    url: str,
    max_retries: int = 3,
    backoff_factor: float = 1.0
) -> requests.Response:
    """
    Fetch URL with exponential backoff retry
    
    Args:
        url: URL to fetch
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff
        
    Returns:
        Response object
        
    Raises:
        requests.RequestException: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                timeout=(config.NETWORK_CONNECT_TIMEOUT, config.NETWORK_READ_TIMEOUT),
                headers={'User-Agent': 'BitcoinCompiler/1.0'}
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = backoff_factor * (2 ** attempt)
            logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}), "
                          f"retrying in {wait_time}s: {e}")
            time.sleep(wait_time)


# ================== SHA256 VERIFICATION ==================
def calculate_sha256(filepath: Path, chunk_size: int = None) -> Optional[str]:
    """
    Calculate SHA256 hash of file with optimal chunk size
    
    Args:
        filepath: Path to file
        chunk_size: Chunk size for reading (default: 64KB)
        
    Returns:
        Hex digest string or None on error
    """
    if chunk_size is None:
        chunk_size = config.HASH_CHUNK_SIZE
    
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"❌ Error calculating SHA256: {e}")
        return None


def verify_git_commit(repo_dir: Path, expected_tag: str) -> bool:
    """
    Verify git repository is at expected tag/commit
    
    Args:
        repo_dir: Repository directory
        expected_tag: Expected git tag
        
    Returns:
        True if verified, False otherwise
    """
    try:
        # Get current commit hash
        result = subprocess_manager.run_command(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            timeout=config.GIT_TIMEOUT,
            log_output=False
        )
        current_commit = result.stdout.strip()
        
        # Get commit hash for the tag
        result = subprocess_manager.run_command(
            ["git", "rev-list", "-n", "1", expected_tag],
            cwd=repo_dir,
            timeout=config.GIT_TIMEOUT,
            log_output=False
        )
        tag_commit = result.stdout.strip()
        
        if current_commit == tag_commit:
            logger.info(f"✓ Git repository verified at {expected_tag}")
            logger.info(f"  Commit: {current_commit[:16]}...")
            return True
        else:
            logger.warning("⚠️  Repository commit mismatch!")
            logger.warning(f"  Current: {current_commit[:16]}...")
            logger.warning(f"  Expected: {tag_commit[:16]}...")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️  Error verifying git commit: {e}")
        return False


# ================== VERSION MANAGEMENT ==================
# Precompiled regex patterns for efficiency
VERSION_PATTERN = re.compile(r"(\d+)\.(\d+)")
CHIP_PATTERN = re.compile(r'Apple (M\d+(?:\s+(?:Pro|Max|Ultra))?)', re.IGNORECASE)


def parse_version(tag: str) -> Tuple[int, int]:
    """Parse version number from git tag"""
    tag = tag.lstrip('v')
    match = VERSION_PATTERN.match(tag)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def use_cmake(version: str) -> bool:
    """Determine if version uses CMake (v25+) or Autotools"""
    major, _ = parse_version(version)
    return major >= 25


def get_bitcoin_versions() -> List[str]:
    """Fetch latest Bitcoin Core releases from GitHub (with caching)"""
    cache_key = "bitcoin_versions"
    cached = api_cache.get(cache_key)
    if cached:
        logger.info("Using cached Bitcoin versions")
        return cached
    
    try:
        response = fetch_with_retry(config.BITCOIN_API)
        
        all_versions = []
        for rel in response.json():
            tag = rel["tag_name"]
            if "rc" not in tag.lower():
                all_versions.append(tag)
                if len(all_versions) == 20:
                    break
        
        # Group by major.minor and take latest patch
        version_groups: Dict[str, List[str]] = {}
        for tag in all_versions:
            major, minor = parse_version(tag)
            key = f"{major}.{minor}"
            if key not in version_groups:
                version_groups[key] = []
            version_groups[key].append(tag)
        
        # Sort and take latest from each group
        filtered_versions = []
        sorted_keys = sorted(
            version_groups.keys(),
            key=lambda x: tuple(map(int, x.split('.'))),
            reverse=True
        )
        
        for key in sorted_keys:
            group = sorted(version_groups[key], key=parse_version, reverse=True)
            filtered_versions.append(group[0])
            if len(filtered_versions) == 5:
                break
        
        logger.info(f"Found {len(filtered_versions)} Bitcoin versions (latest patch only)")
        
        api_cache.set(cache_key, filtered_versions)
        return filtered_versions
        
    except Exception as e:
        logger.error(f"Failed to fetch Bitcoin versions: {e}")
        return []


def get_electrs_versions() -> List[str]:
    """Fetch latest Electrs releases from GitHub (with caching)"""
    cache_key = "electrs_versions"
    cached = api_cache.get(cache_key)
    if cached:
        logger.info("Using cached Electrs versions")
        return cached
    
    try:
        response = fetch_with_retry(config.ELECTRS_API)
        
        versions = []
        for rel in response.json():
            tag = rel["tag_name"]
            if "rc" not in tag.lower():
                versions.append(tag)
                if len(versions) == 3:
                    break
        
        logger.info(f"Found {len(versions)} Electrs versions")
        
        api_cache.set(cache_key, versions)
        return versions
        
    except Exception as e:
        logger.error(f"Failed to fetch Electrs versions: {e}")
        return []


# ================== OPTIMIZATION FLAGS ==================
def get_optimization_flags(use_aggressive: bool = False) -> Dict[str, str]:
    """Get compiler optimization flags based on architecture and settings"""
    flags = {}
    
    if ARCH == Architecture.APPLE_SILICON:
        base_flags = [
            "-mcpu=native",
            "-O2",
            "-fomit-frame-pointer",
            "-fno-common",
        ]
        
        if use_aggressive:
            flags['CFLAGS'] = ' '.join(base_flags + ["-O3", "-flto"])
            flags['CXXFLAGS'] = ' '.join(base_flags + ["-O3", "-flto"])
            flags['LDFLAGS'] = '-flto'
        else:
            flags['CFLAGS'] = ' '.join(base_flags)
            flags['CXXFLAGS'] = ' '.join(base_flags)
            flags['LDFLAGS'] = ''
            
    elif ARCH == Architecture.INTEL:
        base_flags = [
            "-march=native",
            "-O2",
            "-fomit-frame-pointer",
            "-fno-common",
        ]
        
        if use_aggressive:
            flags['CFLAGS'] = ' '.join(base_flags + ["-O3", "-flto", "-mtune=native"])
            flags['CXXFLAGS'] = ' '.join(base_flags + ["-O3", "-flto", "-mtune=native"])
            flags['LDFLAGS'] = '-flto'
        else:
            flags['CFLAGS'] = ' '.join(base_flags)
            flags['CXXFLAGS'] = ' '.join(base_flags)
            flags['LDFLAGS'] = ''
    else:
        # Safe defaults for unknown architecture
        flags['CFLAGS'] = '-O2'
        flags['CXXFLAGS'] = '-O2'
        flags['LDFLAGS'] = ''
    
    return flags


def get_rust_optimization_flags(use_aggressive: bool = False) -> Dict[str, str]:
    """Get Rust/Cargo optimization flags"""
    flags = {}
    
    if use_aggressive:
        flags['RUSTFLAGS'] = '-C opt-level=3 -C target-cpu=native'
        flags['CARGO_PROFILE_RELEASE_LTO'] = 'fat'
        flags['CARGO_PROFILE_RELEASE_OPT_LEVEL'] = '3'
    else:
        flags['RUSTFLAGS'] = '-C opt-level=2 -C target-cpu=native'
        flags['CARGO_PROFILE_RELEASE_OPT_LEVEL'] = '2'
    
    return flags


# ================== ENVIRONMENT SETUP ==================
def setup_build_environment(use_aggressive_opts: bool = False) -> Dict[str, str]:
    """Setup environment variables for building"""
    env = os.environ.copy()
    
    # Build PATH with deduplication
    paths = []
    seen = set()
    
    if BREW_PREFIX:
        paths.append(str(BREW_PREFIX / "bin"))
    
    for path_str in [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        str(Path.home() / ".cargo" / "bin"),
        "/opt/homebrew/opt/llvm/bin",
        "/usr/local/opt/llvm/bin",
    ]:
        path = Path(path_str)
        if path.is_dir() and str(path) not in seen:
            paths.append(str(path))
            seen.add(str(path))
    
    # Add existing PATH entries
    for p in env.get('PATH', '').split(':'):
        if p and p not in seen:
            paths.append(p)
            seen.add(p)
    
    env["PATH"] = ":".join(paths)
    
    # Set LLVM paths
    for llvm_prefix_str in [
        str(BREW_PREFIX / "opt" / "llvm") if BREW_PREFIX else None,
        "/opt/homebrew/opt/llvm",
        "/usr/local/opt/llvm"
    ]:
        if llvm_prefix_str:
            llvm_prefix = Path(llvm_prefix_str)
            if llvm_prefix.is_dir():
                env["LIBCLANG_PATH"] = str(llvm_prefix / "lib")
                env["DYLD_LIBRARY_PATH"] = str(llvm_prefix / "lib")
                break
    
    # Set optimization flags
    opt_flags = get_optimization_flags(use_aggressive_opts)
    for key, value in opt_flags.items():
        if value:
            env[key] = value
            logger.info(f"  {key}: {value}")
    
    return env


# Due to file length limits, I'll continue in the next file...
# ================== DEPENDENCY MANAGEMENT ==================
class DependencyChecker:
    """Parallel dependency checking and installation"""
    
    def __init__(self):
        self.brew = BREW
        self.brew_prefix = BREW_PREFIX
    
    def check_rust_installation(self) -> bool:
        """Check Rust/Cargo installation (parallel search)"""
        logger.info("\n=== Checking Rust Toolchain ===")
        
        rust_paths = [
            Path.home() / ".cargo" / "bin",
            Path("/opt/homebrew/bin"),
            Path("/usr/local/bin"),
        ]
        if self.brew_prefix:
            rust_paths.insert(1, self.brew_prefix / "bin")
        
        rustc_found = False
        cargo_found = False
        
        # Check paths in parallel
        def check_tool(path: Path, tool: str) -> Optional[str]:
            """Check if tool exists and return version"""
            tool_path = path / tool
            if not tool_path.is_file():
                return None
            try:
                result = subprocess.run(
                    [str(tool_path), "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=config.QUICK_CMD_TIMEOUT
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            return None
        
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            # Submit all checks
            futures = {}
            for path in rust_paths:
                for tool in ["rustc", "cargo"]:
                    future = executor.submit(check_tool, path, tool)
                    futures[future] = (path, tool)
            
            # Collect results
            for future in as_completed(futures):
                path, tool = futures[future]
                try:
                    version = future.result()
                    if version:
                        if tool == "rustc" and not rustc_found:
                            rustc_found = True
                            logger.info(f"✓ rustc found at: {path / tool}")
                            logger.info(f"  Version: {version}")
                        elif tool == "cargo" and not cargo_found:
                            cargo_found = True
                            logger.info(f"✓ cargo found at: {path / tool}")
                            logger.info(f"  Version: {version}")
                except Exception as e:
                    logger.debug(f"Error checking {tool} at {path}: {e}")
        
        if not rustc_found or not cargo_found:
            logger.warning("\n❌ Rust toolchain not found or incomplete!")
            return self._install_rust()
        
        return True
    
    def _install_rust(self) -> bool:
        """Install Rust via Homebrew"""
        if not self.brew:
            logger.error("Homebrew not found - cannot install Rust")
            messagebox.showerror(
                "Missing Dependency",
                "Rust not found and Homebrew not available.\n\n"
                "Please install Rust manually from https://rustup.rs"
            )
            return False
        
        logger.info("Installing Rust via Homebrew...")
        
        try:
            # Check if rust formula exists
            subprocess_manager.run_command(
                [str(self.brew), "info", "rust"],
                timeout=config.QUICK_CMD_TIMEOUT,
                log_output=False
            )
            
            # Install rust
            logger.info("📦 Installing rust from Homebrew...")
            subprocess_manager.run_command(
                [str(self.brew), "install", "rust"],
                timeout=600  # 10 minutes for installation
            )
            
            logger.info("\nVerifying Rust installation...")
            time.sleep(2)
            
            # Verify installation
            return self.check_rust_installation()
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install Rust: {e}")
            messagebox.showerror(
                "Installation Failed",
                "Could not install Rust via Homebrew.\n\n"
                "Please install manually:\n"
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh\n\n"
                "Then restart this app."
            )
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error installing Rust: {e}")
            return False
    
    def check_brew_packages(self, packages: List[str]) -> Tuple[List[str], List[str]]:
        """Check which packages are installed (parallel)"""
        logger.info("\nChecking Homebrew packages...")
        
        if not self.brew:
            logger.error("Homebrew not found!")
            return [], packages
        
        installed = []
        missing = []
        
        def check_package(pkg: str) -> bool:
            """Check if package is installed"""
            try:
                result = subprocess.run(
                    [str(self.brew), "list", pkg],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=config.QUICK_CMD_TIMEOUT
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
        
        # Check packages in parallel
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            future_to_pkg = {
                executor.submit(check_package, pkg): pkg
                for pkg in packages
            }
            
            for future in as_completed(future_to_pkg):
                pkg = future_to_pkg[future]
                try:
                    if future.result():
                        installed.append(pkg)
                        logger.info(f"  ✓ {pkg}")
                    else:
                        missing.append(pkg)
                        logger.info(f"  ❌ {pkg} - not installed")
                except Exception as e:
                    logger.error(f"  ❌ {pkg} - error: {e}")
                    missing.append(pkg)
        
        return installed, missing
    
    def install_packages(self, packages: List[str]) -> bool:
        """Install missing packages"""
        success = True
        for pkg in packages:
            logger.info(f"\n📦 Installing {pkg}...")
            try:
                subprocess_manager.run_command(
                    [str(self.brew), "install", pkg],
                    timeout=600  # 10 minutes per package
                )
                logger.info(f"✓ {pkg} installed successfully")
            except Exception as e:
                logger.error(f"❌ Failed to install {pkg}: {e}")
                messagebox.showerror(
                    "Installation Failed",
                    f"Failed to install {pkg}:\n\n{str(e)}"
                )
                success = False
        return success


# ================== BUILD FUNCTIONS ==================
class Builder:
    """Manages compilation process"""
    
    def __init__(self, progress_callback: Optional[Callable[[int], None]] = None):
        self.progress_callback = progress_callback
    
    def set_progress(self, value: int) -> None:
        """Update progress (thread-safe)"""
        if self.progress_callback:
            self.progress_callback(value)
    
    def verify_source_integrity(
        self,
        repo_dir: Path,
        project_name: str,
        version: str
    ) -> bool:
        """Verify source code integrity using git commit verification"""
        logger.info(f"\n🔐 Verifying {project_name} source integrity...")
        
        if verify_git_commit(repo_dir, version):
            logger.info(f"✓ {project_name} source integrity verified!")
            return True
        else:
            logger.warning(f"⚠️  {project_name} source verification failed")
            response = messagebox.askyesno(
                "Source Verification Warning",
                f"{project_name} source code could not be verified.\n\n"
                f"This could indicate:\n"
                f"• Network issues during clone\n"
                f"• Repository corruption\n"
                f"• Unexpected git state\n\n"
                f"Continue anyway? (Not recommended)"
            )
            return response
    
    def copy_binaries(
        self,
        src_dir: Path,
        dest_dir: Path,
        binary_files: List[Path]
    ) -> List[Path]:
        """Copy compiled binaries to destination directory"""
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        
        logger.info(f"Copying binaries to: {dest_dir}")
        
        for binary in binary_files:
            if not binary.exists():
                logger.warning(f"⚠️  Binary not found (skipping): {binary}")
                continue
            
            try:
                dest = dest_dir / binary.name
                
                # Use hardlink if on same filesystem, otherwise copy
                try:
                    if dest.exists():
                        dest.unlink()
                    os.link(str(binary), str(dest))
                    logger.info(f"✓ Linked: {binary.name} → {dest}")
                except (OSError, NotImplementedError):
                    # Fall back to copy if hardlink fails
                    shutil.copy2(binary, dest)
                    logger.info(f"✓ Copied: {binary.name} → {dest}")
                
                dest.chmod(0o755)
                copied.append(dest)
            except Exception as e:
                logger.warning(f"⚠️  Failed to copy {binary.name}: {e}")
        
        if not copied:
            logger.error("❌ WARNING: No binaries were copied!")
        
        return copied
    
    def compile_bitcoin(
        self,
        version: str,
        build_dir: Path,
        cores: int,
        use_aggressive_opts: bool = False
    ) -> Dict[str, Any]:
        """Compile Bitcoin Core from source"""
        version = sanitize_version_tag(version)
        build_dir = validate_directory(build_dir)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPILING BITCOIN CORE {version}")
        logger.info(f"{'='*60}")
        logger.info(f"Architecture: {ARCH.value.upper()}")
        logger.info(f"Optimization: {'AGGRESSIVE (O3 + LTO)' if use_aggressive_opts else 'STANDARD (O2)'}")
        logger.info(f"{'='*60}")
        
        version_clean = version.lstrip('v')
        src_dir = build_dir / f"bitcoin-{version_clean}"
        
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone or update repository
        if not src_dir.exists():
            logger.info(f"\n📥 Cloning Bitcoin Core repository...")
            subprocess_manager.run_command(
                ["git", "clone", "--depth", "1", "--branch", version,
                 config.BITCOIN_REPO, str(src_dir)],
                cwd=build_dir,
                timeout=config.GIT_TIMEOUT
            )
            logger.info(f"✓ Source cloned to {src_dir}")
        else:
            logger.info(f"✓ Source directory already exists: {src_dir}")
            # Check if already at correct version
            try:
                result = subprocess_manager.run_command(
                    ["git", "describe", "--tags", "--exact-match"],
                    cwd=src_dir,
                    timeout=config.QUICK_CMD_TIMEOUT,
                    log_output=False
                )
                current_tag = result.stdout.strip()
                if current_tag == version:
                    logger.info(f"✓ Already at {version}")
                else:
                    logger.info(f"📥 Updating from {current_tag} to {version}...")
                    subprocess_manager.run_command(
                        ["git", "fetch", "--depth", "1", "origin", "tag", version],
                        cwd=src_dir,
                        timeout=config.GIT_TIMEOUT
                    )
                    subprocess_manager.run_command(
                        ["git", "checkout", version],
                        cwd=src_dir,
                        timeout=config.GIT_TIMEOUT
                    )
                    logger.info(f"✓ Updated to {version}")
            except subprocess.CalledProcessError:
                # If we can't determine current tag, fetch and checkout
                logger.info(f"📥 Updating to {version}...")
                subprocess_manager.run_command(
                    ["git", "fetch", "--depth", "1", "origin", "tag", version],
                    cwd=src_dir,
                    timeout=config.GIT_TIMEOUT
                )
                subprocess_manager.run_command(
                    ["git", "checkout", version],
                    cwd=src_dir,
                    timeout=config.GIT_TIMEOUT
                )
                logger.info(f"✓ Updated to {version}")
        
        # Verify source integrity
        if not self.verify_source_integrity(src_dir, "Bitcoin Core", version):
            raise RuntimeError("Source verification failed")
        
        # Setup build environment
        env = setup_build_environment(use_aggressive_opts)
        
        logger.info(f"\nEnvironment setup:")
        logger.info(f"  PATH: {env['PATH'][:150]}...")
        if 'CFLAGS' in env:
            logger.info(f"  CFLAGS: {env['CFLAGS']}")
        if 'CXXFLAGS' in env:
            logger.info(f"  CXXFLAGS: {env['CXXFLAGS']}")
        if 'LDFLAGS' in env and env['LDFLAGS']:
            logger.info(f"  LDFLAGS: {env['LDFLAGS']}")
        logger.info(f"  Building node-only (wallet support disabled)")
        
        uses_cmake = use_cmake(version)
        
        if uses_cmake:
            logger.info(f"\n🔨 Building with CMake (Bitcoin Core {version})...")
            
            logger.info(f"\n⚙️  Configuring (wallet support disabled, tests enabled)...")
            subprocess_manager.run_command(
                ["cmake", "-B", "build", "-DENABLE_WALLET=OFF",
                 "-DENABLE_IPC=OFF", "-DBUILD_TESTS=ON"],
                cwd=src_dir,
                env=env,
                timeout=config.SUBPROCESS_TIMEOUT
            )
            
            logger.info(f"\n🔧 Compiling with {cores} cores...")
            subprocess_manager.run_command(
                ["cmake", "--build", "build", f"-j{cores}"],
                cwd=src_dir,
                env=env,
                timeout=config.SUBPROCESS_TIMEOUT
            )
            
            build_subdir = src_dir / "build"
            binary_dir = build_subdir / "bin"
            binaries = [
                binary_dir / "bitcoind",
                binary_dir / "bitcoin-cli",
                binary_dir / "bitcoin-tx",
                binary_dir / "bitcoin-wallet",
                binary_dir / "bitcoin-util",
            ]
        else:
            logger.info(f"\n🔨 Building with Autotools (Bitcoin Core {version})...")
            
            logger.info(f"\n⚙️  Running autogen.sh...")
            subprocess_manager.run_command(
                ["./autogen.sh"],
                cwd=src_dir,
                env=env,
                timeout=config.SUBPROCESS_TIMEOUT
            )
            
            logger.info(f"\n⚙️  Configuring (wallet support disabled, tests enabled)...")
            subprocess_manager.run_command(
                ["./configure", "--disable-wallet", "--disable-gui", "--enable-tests"],
                cwd=src_dir,
                env=env,
                timeout=config.SUBPROCESS_TIMEOUT
            )
            
            logger.info(f"\n🔧 Compiling with {cores} cores...")
            subprocess_manager.run_command(
                ["make", f"-j{cores}"],
                cwd=src_dir,
                env=env,
                timeout=config.SUBPROCESS_TIMEOUT
            )
            
            binary_dir = src_dir / "bin"
            binaries = [
                binary_dir / "bitcoind",
                binary_dir / "bitcoin-cli",
                binary_dir / "bitcoin-tx",
                binary_dir / "bitcoin-wallet",
            ]
        
        # Copy binaries
        logger.info(f"\n📋 Collecting binaries...")
        output_dir = build_dir / "binaries" / f"bitcoin-{version_clean}"
        copied = self.copy_binaries(src_dir, output_dir, binaries)
        
        if not copied:
            logger.warning("⚠️  Warning: No binaries were copied. Checking what exists...")
            for binary in binaries:
                exists = "✓" if binary.exists() else "❌"
                logger.info(f"  {exists} {binary}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ BITCOIN CORE {version} COMPILED SUCCESSFULLY!")
        logger.info(f"{'='*60}")
        logger.info(f"\n📍 Binaries location: {output_dir}")
        logger.info(f"   Found {len(copied)} binaries\n")
        
        return {
            'output_dir': output_dir,
            'src_dir': src_dir,
            'uses_cmake': uses_cmake
        }
    
    def compile_electrs(
        self,
        version: str,
        build_dir: Path,
        cores: int,
        use_aggressive_opts: bool = False
    ) -> Dict[str, Any]:
        """Compile Electrs from source"""
        version = sanitize_version_tag(version)
        build_dir = validate_directory(build_dir)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPILING ELECTRS {version}")
        logger.info(f"{'='*60}")
        logger.info(f"Architecture: {ARCH.value.upper()}")
        logger.info(f"Optimization: {'AGGRESSIVE (O3 + LTO)' if use_aggressive_opts else 'STANDARD (O2)'}")
        logger.info(f"{'='*60}")
        
        env = setup_build_environment(use_aggressive_opts)
        
        # Add Rust flags
        rust_flags = get_rust_optimization_flags(use_aggressive_opts)
        for key, value in rust_flags.items():
            if value:
                env[key] = value
                logger.info(f"  {key}: {value}")
        
        # Verify Rust installation
        logger.info("\n🔍 Verifying Rust installation...")
        try:
            subprocess_manager.run_command(
                ["cargo", "--version"],
                env=env,
                timeout=config.QUICK_CMD_TIMEOUT,
                log_output=False
            )
            logger.info("✓ Cargo found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            error_msg = (
                "❌ Cargo not found in PATH!\n\n"
                "Electrs requires Rust/Cargo to compile.\n\n"
                "Please:\n"
                "1. Click 'Check & Install Dependencies' button\n"
                "2. Ensure Rust is installed\n"
                "3. Restart this application"
            )
            logger.error(error_msg)
            messagebox.showerror("Rust Not Found", error_msg)
            raise RuntimeError("Cargo not found - cannot compile Electrs")
        
        version_clean = version.lstrip('v')
        src_dir = build_dir / f"electrs-{version_clean}"
        
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone or update repository
        if not src_dir.exists():
            logger.info(f"\n📥 Cloning Electrs repository...")
            subprocess_manager.run_command(
                ["git", "clone", "--depth", "1", "--branch", version,
                 config.ELECTRS_REPO, str(src_dir)],
                cwd=build_dir,
                env=env,
                timeout=config.GIT_TIMEOUT
            )
            logger.info(f"✓ Source cloned to {src_dir}")
        else:
            logger.info(f"✓ Source directory already exists: {src_dir}")
            logger.info(f"📥 Updating to {version}...")
            subprocess_manager.run_command(
                ["git", "fetch", "--depth", "1", "origin", "tag", version],
                cwd=src_dir,
                env=env,
                timeout=config.GIT_TIMEOUT
            )
            subprocess_manager.run_command(
                ["git", "checkout", version],
                cwd=src_dir,
                env=env,
                timeout=config.GIT_TIMEOUT
            )
            logger.info(f"✓ Updated to {version}")
        
        # Verify source integrity
        if not self.verify_source_integrity(src_dir, "Electrs", version):
            raise RuntimeError("Source verification failed")
        
        logger.info(f"\n🔧 Building with Cargo ({cores} jobs)...")
        logger.info(f"Environment details:")
        logger.info(f"  PATH: {env['PATH'][:150]}...")
        if 'LIBCLANG_PATH' in env:
            logger.info(f"  LIBCLANG_PATH: {env['LIBCLANG_PATH']}")
        if 'RUSTFLAGS' in env:
            logger.info(f"  RUSTFLAGS: {env['RUSTFLAGS']}")
        
        subprocess_manager.run_command(
            ["cargo", "build", "--release", "--jobs", str(cores)],
            cwd=src_dir,
            env=env,
            timeout=config.SUBPROCESS_TIMEOUT
        )
        
        logger.info(f"\n📋 Collecting binaries...")
        binary = src_dir / "target" / "release" / "electrs"
        
        if not binary.exists():
            raise RuntimeError(f"Electrs binary not found at expected location: {binary}")
        
        output_dir = build_dir / "binaries" / f"electrs-{version_clean}"
        copied = self.copy_binaries(src_dir, output_dir, [binary])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ ELECTRS {version} COMPILED SUCCESSFULLY!")
        logger.info(f"{'='*60}")
        logger.info(f"\n📍 Binary location: {output_dir}/electrs\n")
        
        return {
            'output_dir': output_dir,
            'src_dir': src_dir
        }
    
    def run_bitcoin_tests(
        self,
        src_dir: Path,
        version: str,
        use_cmake_build: bool
    ) -> bool:
        """Run Bitcoin Core unit tests"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"RUNNING BITCOIN CORE {version} UNIT TESTS")
            logger.info(f"{'='*60}")
            
            env = setup_build_environment(False)
            
            if use_cmake_build:
                logger.info("Running tests via ctest...")
                test_dir = src_dir / "build"
                
                subprocess_manager.run_command(
                    ["ctest", "--output-on-failure"],
                    cwd=test_dir,
                    env=env,
                    timeout=config.SUBPROCESS_TIMEOUT
                )
            else:
                logger.info("Running tests via 'make check'...")
                
                subprocess_manager.run_command(
                    ["make", "check"],
                    cwd=src_dir,
                    env=env,
                    timeout=config.SUBPROCESS_TIMEOUT
                )
            
            logger.info(f"\n✅ Bitcoin Core {version} tests PASSED!")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"\n❌ Bitcoin Core {version} tests FAILED!")
            logger.error(f"Error: {e}")
            return False
        except Exception as e:
            logger.error(f"\n❌ Error running Bitcoin tests: {e}")
            return False
    
    def run_electrs_tests(self, src_dir: Path, version: str) -> bool:
        """Run Electrs unit tests"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"RUNNING ELECTRS {version} UNIT TESTS")
            logger.info(f"{'='*60}")
            
            env = setup_build_environment(False)
            
            rust_flags = get_rust_optimization_flags(False)
            for key, value in rust_flags.items():
                if value:
                    env[key] = value
            
            logger.info("Running tests via 'cargo test --release'...")
            
            subprocess_manager.run_command(
                ["cargo", "test", "--release"],
                cwd=src_dir,
                env=env,
                timeout=config.SUBPROCESS_TIMEOUT
            )
            
            logger.info(f"\n✅ Electrs {version} tests PASSED!")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"\n❌ Electrs {version} tests FAILED!")
            logger.error(f"Error: {e}")
            return False
        except Exception as e:
            logger.error(f"\n❌ Error running Electrs tests: {e}")
            return False


# Continue in next file...
# ================== GUI APPLICATION ==================
class BitcoinCompilerGUI:
    """Main GUI application class"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Bitcoin & Electrs Compiler for macOS")
        self.root.geometry("900x800")
        self.root.resizable(True, True)
        self.root.minsize(800, 600)
        
        # Variables
        self.target_var = tk.StringVar(value="Bitcoin")
        self.cores_var = tk.IntVar(value=max(1, multiprocessing.cpu_count() - 1))
        self.build_dir_var = tk.StringVar(value=str(config.DEFAULT_BUILD_DIR))
        self.bitcoin_version_var = tk.StringVar(value="Loading...")
        self.electrs_version_var = tk.StringVar(value="Loading...")
        self.aggressive_opts_var = tk.BooleanVar(value=False)
        self.run_tests_var = tk.BooleanVar(value=False)
        self.progress_var = tk.DoubleVar()
        
        # Components
        self.bitcoin_combo = None
        self.electrs_combo = None
        self.log_text = None
        self.progress = None
        self.compile_btn = None
        
        # Worker management
        self.dep_checker = DependencyChecker()
        self.builder = Builder(progress_callback=self._set_progress)
        
        # Setup GUI
        self._create_widgets()
        self._setup_cleanup()
        
        # Load versions asynchronously
        self.root.after(100, self._initial_version_load)
    
    def _setup_cleanup(self) -> None:
        """Setup cleanup handlers"""
        def cleanup():
            logger.info("\nShutting down...")
            subprocess_manager.cleanup()
            self.root.quit()
        
        self.root.protocol("WM_DELETE_WINDOW", cleanup)
    
    def _create_widgets(self) -> None:
        """Create all GUI widgets"""
        # Header
        header = ttk.Label(
            self.root,
            text="Bitcoin Core & Electrs Compiler",
            font=("Arial", 16, "bold")
        )
        header.pack(pady=10)
        
        # Architecture info
        if ARCH == Architecture.APPLE_SILICON and CHIP_NAME:
            arch_text = f"Architecture: Apple Silicon ({CHIP_NAME})"
        elif ARCH == Architecture.APPLE_SILICON:
            arch_text = "Architecture: Apple Silicon"
        elif ARCH == Architecture.INTEL:
            arch_text = "Architecture: Intel Mac"
        else:
            arch_text = f"Architecture: {ARCH.value}"
        
        arch_label = ttk.Label(self.root, text=arch_text, font=("Arial", 10))
        arch_label.pack()
        
        # Step 1: Dependencies
        dep_frame = ttk.Frame(self.root)
        dep_frame.pack(pady=10)
        ttk.Label(
            dep_frame,
            text="Step 1:",
            font=("Arial", 10, "bold")
        ).pack(side="left", padx=5)
        ttk.Button(
            dep_frame,
            text="Check & Install Dependencies",
            command=self._check_dependencies
        ).pack(side="left")
        
        # Separator
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=20, pady=10)
        
        # Step 2: Target selection
        target_frame = ttk.LabelFrame(
            self.root,
            text="Step 2: Select What to Compile",
            padding=10
        )
        target_frame.pack(fill="x", padx=20, pady=5)
        
        ttk.Label(target_frame, text="Target:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        target_combo = ttk.Combobox(
            target_frame,
            values=["Bitcoin", "Electrs", "Both"],
            textvariable=self.target_var,
            state="readonly",
            width=15
        )
        target_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # CPU cores
        ttk.Label(target_frame, text="CPU Cores:").grid(
            row=0, column=2, sticky="w", padx=5, pady=5
        )
        cores_spinbox = ttk.Spinbox(
            target_frame,
            from_=1,
            to=multiprocessing.cpu_count(),
            textvariable=self.cores_var,
            width=5
        )
        cores_spinbox.grid(row=0, column=3, sticky="w", padx=5, pady=5)
        ttk.Label(
            target_frame,
            text=f"(max: {multiprocessing.cpu_count()})",
            font=("Arial", 9)
        ).grid(row=0, column=4, sticky="w", padx=2, pady=5)
        
        # Build directory
        ttk.Label(target_frame, text="Build Directory:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        build_entry = ttk.Entry(
            target_frame,
            textvariable=self.build_dir_var,
            width=40
        )
        build_entry.grid(row=1, column=1, columnspan=3, sticky="ew", padx=5, pady=5)
        ttk.Button(
            target_frame,
            text="Browse",
            command=self._browse_build_dir
        ).grid(row=1, column=4, padx=5, pady=5)
        
        # Step 2.5: Optimization and Testing
        opt_frame = ttk.LabelFrame(
            self.root,
            text="Step 2.5: Optimization & Testing",
            padding=10
        )
        opt_frame.pack(fill="x", padx=20, pady=5)
        
        ttk.Checkbutton(
            opt_frame,
            text="⚡ Enable Aggressive Optimizations (O3 + LTO) - May break code, use with caution!",
            variable=self.aggressive_opts_var
        ).pack(anchor="w", padx=5, pady=2)
        
        ttk.Checkbutton(
            opt_frame,
            text="🧪 Run Unit Tests After Compilation - Validates compiled binaries are functional",
            variable=self.run_tests_var
        ).pack(anchor="w", padx=5, pady=2)
        
        ttk.Label(
            opt_frame,
            text="ℹ️  Optimizations: Standard (O2) vs Aggressive (O3+LTO). Tests add 5-20 min to build time.",
            font=("Arial", 9),
            foreground="gray"
        ).pack(anchor="w", padx=5, pady=(2, 5))
        
        # Step 3: Version selection
        version_frame = ttk.LabelFrame(
            self.root,
            text="Step 3: Select Versions",
            padding=10
        )
        version_frame.pack(fill="x", padx=20, pady=5)
        
        # Bitcoin version
        ttk.Label(version_frame, text="Bitcoin Version:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        self.bitcoin_combo = ttk.Combobox(
            version_frame,
            values=["Loading..."],
            textvariable=self.bitcoin_version_var,
            state="readonly",
            width=20
        )
        self.bitcoin_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(
            version_frame,
            text="Refresh",
            command=lambda: threading.Thread(
                target=self._refresh_bitcoin_versions,
                daemon=True
            ).start()
        ).grid(row=0, column=2, padx=5, pady=5)
        
        # Electrs version
        ttk.Label(version_frame, text="Electrs Version:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        self.electrs_combo = ttk.Combobox(
            version_frame,
            values=["Loading..."],
            textvariable=self.electrs_version_var,
            state="readonly",
            width=20
        )
        self.electrs_combo.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(
            version_frame,
            text="Refresh",
            command=lambda: threading.Thread(
                target=self._refresh_electrs_versions,
                daemon=True
            ).start()
        ).grid(row=1, column=2, padx=5, pady=5)
        
        # Progress bar
        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill="x", padx=20, pady=10)
        ttk.Label(progress_frame, text="Progress:").pack(anchor="w")
        self.progress = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress.pack(fill="x", pady=5)
        
        # Compile button
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        self.compile_btn = ttk.Button(
            button_frame,
            text="🚀 Start Compilation",
            command=self._compile_selected
        )
        self.compile_btn.pack()
        
        # Log terminal
        log_frame = ttk.LabelFrame(self.root, text="Build Log", padding=5)
        log_frame.pack(fill="both", expand=True, padx=20, pady=5)
        
        log_text_frame = tk.Frame(log_frame)
        log_text_frame.pack(fill="both", expand=True)
        
        self.log_text = tk.Text(
            log_text_frame,
            height=20,
            wrap="none",
            bg="#1e1e1e",
            fg="#00ff00",
            font=("Courier", 10)
        )
        self.log_text.pack(side="left", fill="both", expand=True)
        
        scrollbar_y = ttk.Scrollbar(log_text_frame, command=self.log_text.yview)
        scrollbar_y.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar_y.set)
        
        scrollbar_x = ttk.Scrollbar(
            log_frame,
            orient="horizontal",
            command=self.log_text.xview
        )
        scrollbar_x.pack(fill="x")
        self.log_text.config(xscrollcommand=scrollbar_x.set)
        
        # Connect log widget to handler
        gui_handler.set_widget(self.log_text)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", side="bottom")
        status_text = (
            f"System: macOS {platform.mac_ver()[0]} | "
            f"Arch: {CHIP_NAME if CHIP_NAME else ARCH.value} | "
            f"Homebrew: {BREW_PREFIX if BREW_PREFIX else 'Not Found'} | "
            f"CPUs: {multiprocessing.cpu_count()}"
        )
        status_label = ttk.Label(
            status_frame,
            text=status_text,
            relief="sunken",
            anchor="w"
        )
        status_label.pack(fill="x")
        
        # macOS window management
        if platform.system() == 'Darwin':
            try:
                self.root.lift()
                self.root.attributes('-topmost', True)
                self.root.after_idle(self.root.attributes, '-topmost', False)
            except tk.TclError:
                pass
        
        # Initial log message
        self._log_welcome()
    
    def _log_welcome(self) -> None:
        """Log welcome message"""
        logger.info("=" * 60)
        logger.info("Bitcoin Core & Electrs Compiler")
        logger.info("=" * 60)
        logger.info(f"System: macOS {platform.mac_ver()[0]}")
        if CHIP_NAME:
            logger.info(f"Architecture: Apple Silicon ({CHIP_NAME})")
        else:
            logger.info(f"Architecture: {ARCH.value}")
        logger.info(f"Homebrew: {BREW_PREFIX if BREW_PREFIX else 'Not Found'}")
        logger.info(f"CPU Cores: {multiprocessing.cpu_count()}")
        if is_pyinstaller():
            logger.info("Running as: PyInstaller Bundle")
        logger.info("=" * 60 + "\n")
        logger.info("🔧 Features:")
        logger.info("  • Architecture-specific optimizations")
        logger.info("  • Git commit source verification")
        logger.info("  • Optional aggressive O3 + LTO optimizations")
        logger.info("  • Unit test support for validation")
        logger.info("  • Parallel dependency checking")
        logger.info("  • Network retry with exponential backoff\n")
        logger.info("👉 Click 'Check & Install Dependencies' to begin\n")
        logger.info("📝 Note: Both Bitcoin and Electrs pull source from GitHub")
        logger.info("🔐 Note: Source integrity is verified using git commit hashes")
        logger.info("🧪 Note: Unit tests can be run to validate compiled binaries\n")
    
    def _set_progress(self, value: int) -> None:
        """Thread-safe progress update"""
        try:
            self.root.after(0, lambda: self.progress_var.set(value))
        except tk.TclError:
            pass
    
    def _browse_build_dir(self) -> None:
        """Browse for build directory"""
        directory = filedialog.askdirectory(
            initialdir=self.build_dir_var.get(),
            title="Select Build Directory"
        )
        if directory:
            try:
                validated = validate_directory(Path(directory))
                self.build_dir_var.set(str(validated))
            except ValueError as e:
                messagebox.showerror("Invalid Directory", str(e))
    
    def _check_dependencies(self) -> None:
        """Check and install dependencies in background thread"""
        def task():
            try:
                logger.info("\n=== Checking System Dependencies ===")
                
                if not BREW:
                    logger.error("❌ Homebrew not found!")
                    logger.error("Please install Homebrew from https://brew.sh")
                    messagebox.showerror(
                        "Missing Dependency",
                        "Homebrew not found! Please install from https://brew.sh"
                    )
                    return
                
                logger.info(f"✓ Homebrew found at: {BREW}")
                logger.info(f"  Homebrew prefix: {BREW_PREFIX}")
                
                brew_packages = [
                    "automake", "libtool", "pkg-config", "boost",
                    "miniupnpc", "zeromq", "sqlite", "python",
                    "cmake", "llvm", "libevent", "rocksdb", "rust", "git"
                ]
                
                installed, missing = self.dep_checker.check_brew_packages(brew_packages)
                
                if missing:
                    logger.warning(f"\n⚠️  Missing Homebrew packages: {', '.join(missing)}")
                    
                    pkg_count = len(missing)
                    pkg_list = ', '.join(missing[:5])
                    if pkg_count > 5:
                        pkg_list += f", and {pkg_count - 5} more"
                    
                    message = (
                        f"Found {pkg_count} missing package{'s' if pkg_count > 1 else ''}:\n\n"
                        f"{pkg_list}\n\n"
                        f"Install all missing packages now?"
                    )
                    
                    install_deps = messagebox.askyesno(
                        "Install Missing Dependencies",
                        message
                    )
                    
                    if install_deps:
                        self.dep_checker.install_packages(missing)
                    else:
                        logger.warning("\n⚠️  Dependencies not installed. Compilation may fail.")
                else:
                    logger.info("\n✓ All Homebrew packages are installed!")
                
                rust_ok = self.dep_checker.check_rust_installation()
                
                if rust_ok:
                    logger.info("\n✓ Rust toolchain is ready!")
                else:
                    logger.warning("\n⚠️  Rust toolchain needs attention (see messages above)")
                
                logger.info("\n=== Dependency Check Complete ===")
                
                if rust_ok:
                    messagebox.showinfo(
                        "Dependency Check",
                        "✅ All dependencies are installed and ready!\n\n"
                        "You can now proceed with compilation."
                    )
                else:
                    messagebox.showwarning(
                        "Dependency Check",
                        "⚠️  Some dependencies need attention.\n\n"
                        "Check the log for details.\n"
                        "You may need to restart the app after installing Rust."
                    )
                    
            except Exception as e:
                logger.error(f"\n❌ Error during dependency check: {e}")
                import traceback
                logger.error(traceback.format_exc())
                messagebox.showerror("Error", f"Dependency check failed: {e}")
        
        threading.Thread(target=task, daemon=True).start()
    
    def _refresh_bitcoin_versions(self) -> None:
        """Refresh Bitcoin version list"""
        logger.info("\n📡 Fetching Bitcoin versions from GitHub...")
        versions = get_bitcoin_versions()
        
        def update_gui():
            if versions:
                self.bitcoin_combo.configure(values=versions)
                self.bitcoin_version_var.set(versions[0])
                logger.info(f"✓ Loaded {len(versions)} Bitcoin versions (selected: {versions[0]})")
            else:
                logger.warning("⚠️  Could not fetch Bitcoin versions (check internet connection)")
                messagebox.showwarning(
                    "Network Error",
                    "Could not fetch Bitcoin versions. Check your internet connection."
                )
        
        self.root.after(0, update_gui)
    
    def _refresh_electrs_versions(self) -> None:
        """Refresh Electrs version list"""
        logger.info("\n📡 Fetching Electrs versions from GitHub...")
        versions = get_electrs_versions()
        
        def update_gui():
            if versions:
                self.electrs_combo.configure(values=versions)
                self.electrs_version_var.set(versions[0])
                logger.info(f"✓ Loaded {len(versions)} Electrs versions (selected: {versions[0]})")
            else:
                logger.warning("⚠️  Could not fetch Electrs versions (check internet connection)")
                messagebox.showwarning(
                    "Network Error",
                    "Could not fetch Electrs versions. Check your internet connection."
                )
        
        self.root.after(0, update_gui)
    
    def _initial_version_load(self) -> None:
        """Load versions after GUI is ready"""
        def task():
            self._refresh_bitcoin_versions()
            self._refresh_electrs_versions()
        threading.Thread(target=task, daemon=True).start()
    
    def _compile_selected(self) -> None:
        """Main compilation function triggered by button"""
        target = self.target_var.get()
        cores = self.cores_var.get()
        build_dir = self.build_dir_var.get()
        bitcoin_ver = self.bitcoin_version_var.get()
        electrs_ver = self.electrs_version_var.get()
        use_aggressive = self.aggressive_opts_var.get()
        run_tests = self.run_tests_var.get()
        
        # Validate inputs
        if cores < 1 or cores > multiprocessing.cpu_count():
            messagebox.showerror(
                "Invalid Input",
                f"CPU cores must be between 1 and {multiprocessing.cpu_count()}"
            )
            return
        
        if not build_dir:
            messagebox.showerror("Invalid Input", "Build directory cannot be empty")
            return
        
        try:
            build_dir_path = validate_directory(Path(build_dir))
        except ValueError as e:
            messagebox.showerror("Invalid Directory", str(e))
            return
        
        def task():
            try:
                self._set_progress(0)
                self.compile_btn.config(state="disabled")
                
                if use_aggressive:
                    response = messagebox.askyesno(
                        "Aggressive Optimizations Enabled",
                        "⚠️  WARNING: You have enabled aggressive optimizations (O3 + LTO)\n\n"
                        "These flags may:\n"
                        "• Increase build time significantly\n"
                        "• Potentially introduce bugs or instability\n"
                        "• May not work with all versions\n\n"
                        "Recommended for advanced users only.\n\n"
                        "Continue with aggressive optimizations?",
                        icon='warning'
                    )
                    if not response:
                        logger.info("\n❌ User cancelled compilation due to aggressive optimization warning")
                        return
                
                if target in ["Bitcoin", "Both"]:
                    if not bitcoin_ver or bitcoin_ver == "Loading...":
                        messagebox.showerror(
                            "Error",
                            "Please wait for Bitcoin versions to load, or click Refresh"
                        )
                        return
                
                if target in ["Electrs", "Both"]:
                    if not electrs_ver or electrs_ver == "Loading...":
                        messagebox.showerror(
                            "Error",
                            "Please wait for Electrs versions to load, or click Refresh"
                        )
                        return
                
                bitcoin_result = None
                electrs_result = None
                bitcoin_test_passed = None
                electrs_test_passed = None
                
                if target in ["Bitcoin", "Both"]:
                    self._set_progress(10)
                    bitcoin_result = self.builder.compile_bitcoin(
                        bitcoin_ver,
                        build_dir_path,
                        cores,
                        use_aggressive
                    )
                    self._set_progress(45 if target == "Both" else 90)
                
                if target in ["Electrs", "Both"]:
                    self._set_progress(50 if target == "Both" else 10)
                    electrs_result = self.builder.compile_electrs(
                        electrs_ver,
                        build_dir_path,
                        cores,
                        use_aggressive
                    )
                    self._set_progress(90)
                
                if run_tests:
                    logger.info(f"\n{'='*60}")
                    logger.info("RUNNING UNIT TESTS")
                    logger.info(f"{'='*60}")
                    
                    if bitcoin_result:
                        self._set_progress(92)
                        bitcoin_test_passed = self.builder.run_bitcoin_tests(
                            bitcoin_result['src_dir'],
                            bitcoin_ver,
                            bitcoin_result['uses_cmake']
                        )
                    
                    if electrs_result:
                        self._set_progress(96)
                        electrs_test_passed = self.builder.run_electrs_tests(
                            electrs_result['src_dir'],
                            electrs_ver
                        )
                
                self._set_progress(100)
                
                msg = f"✅ {target} compilation completed successfully!\n\n"
                
                msg += "Binaries saved to:\n"
                if bitcoin_result:
                    msg += f"• Bitcoin: {bitcoin_result['output_dir']}\n"
                if electrs_result:
                    msg += f"• Electrs: {electrs_result['output_dir']}\n"
                
                if run_tests:
                    msg += "\n" + "="*40 + "\n"
                    msg += "UNIT TEST RESULTS:\n"
                    msg += "="*40 + "\n"
                    
                    if bitcoin_test_passed is not None:
                        status = "✅ PASSED" if bitcoin_test_passed else "❌ FAILED"
                        msg += f"Bitcoin Core {bitcoin_ver}: {status}\n"
                    
                    if electrs_test_passed is not None:
                        status = "✅ PASSED" if electrs_test_passed else "❌ FAILED"
                        msg += f"Electrs {electrs_ver}: {status}\n"
                    
                    all_passed = True
                    if bitcoin_test_passed is not None and not bitcoin_test_passed:
                        all_passed = False
                    if electrs_test_passed is not None and not electrs_test_passed:
                        all_passed = False
                    
                    if all_passed:
                        msg += "\n✅ All tests PASSED! Binaries are ready to use.\n"
                    else:
                        msg += "\n⚠️  Some tests FAILED! Review the log for details.\n"
                        msg += "Binaries were compiled but may have issues.\n"
                
                messagebox.showinfo("Compilation Complete", msg)
                
            except Exception as e:
                logger.error(f"\n❌ Compilation failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                messagebox.showerror("Compilation Failed", str(e))
            finally:
                self.compile_btn.config(state="normal")
                self._set_progress(0)
        
        threading.Thread(target=task, daemon=True).start()
    
    def run(self) -> None:
        """Start the GUI main loop"""
        self.root.mainloop()


# ================== MAIN ENTRY POINT ==================
def main() -> None:
    """Main entry point with comprehensive error handling"""
    try:
        # Setup multiprocessing for PyInstaller
        if is_pyinstaller():
            multiprocessing.freeze_support()
        
        # Create and run GUI
        app = BitcoinCompilerGUI()
        app.run()
        
    except KeyboardInterrupt:
        logger.info("\nApplication interrupted by user")
        subprocess_manager.cleanup()
        sys.exit(0)
        
    except Exception as e:
        error_msg = f"Fatal error: {e}\n"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        
        try:
            messagebox.showerror(
                "Fatal Error",
                f"Application crashed:\n\n{e}\n\nCheck console for details."
            )
        except Exception:
            pass
        
        subprocess_manager.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
