import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import threading
import requests
import os
import multiprocessing
import shutil
import re
import sys

# ================== CONFIG ==================
BITCOIN_API = "https://api.github.com/repos/bitcoin/bitcoin/releases"
ELECTRS_API = "https://api.github.com/repos/romanz/electrs/releases"
DEFAULT_BUILD_DIR = os.path.expanduser("~/Downloads")

# ================== HOMEBREW DETECTION ==================
def find_brew():
	paths = ["/opt/homebrew/bin/brew", "/usr/local/bin/brew"]
	for p in paths:
		if os.path.isfile(p):
			return p
	return None

BREW = find_brew()

# ================== GUI-SAFE HELPERS ==================
def log(msg):
	log_text.after(0, lambda: (
		log_text.insert("end", msg),
		log_text.see("end")
	))

def set_progress(val):
	progress.after(0, lambda: progress_var.set(val))

def run_command(cmd, cwd=None, progress_cb=None, env=None):
	log(f"\n$ {cmd}\n")
	if env is None:
		env = os.environ.copy()
		env["PATH"] = f"{os.path.expanduser('~/.cargo/bin')}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

	process = subprocess.Popen(
		cmd,
		shell=True,
		cwd=cwd,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		bufsize=1,
		env=env
	)

	for line in process.stdout:
		log(line)
		if progress_cb:
			progress_cb(line)

	process.wait()
	if process.returncode != 0:
		raise RuntimeError(cmd)

# ================== VERSION LOGIC ==================
def parse_version(tag):
	m = re.match(r"v?(\d+)\.(\d+)", tag)
	return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

def use_cmake(version):
	major, _ = parse_version(version)
	return major >= 25

def get_bitcoin_versions():
	r = requests.get(BITCOIN_API, timeout=10)
	r.raise_for_status()
	versions = []
	for rel in r.json():
		tag = rel["tag_name"]
		if "rc" in tag.lower():
			continue
		versions.append(tag)
		if len(versions) == 10:
			break
	return versions

def get_electrs_versions():
	r = requests.get(ELECTRS_API, timeout=10)
	r.raise_for_status()
	versions = []
	for rel in r.json():
		tag = rel["tag_name"]
		if "rc" in tag.lower():
			continue
		versions.append(tag)
		if len(versions) == 10:
			break
	return versions

# ================== DEPENDENCY CHECKER ==================
def check_dependencies():
	def task():
		try:
			log("Checking system dependencies...\n")

			if not BREW:
				log("Homebrew not found. Please install from https://brew.sh\n")
				messagebox.showerror("Missing Dependency", "Homebrew not found!")
				return

			brew_packages = [
				"automake", "libtool", "pkg-config", "boost",
				"berkeley-db@4", "openssl", "miniupnpc",
				"zeromq", "sqlite", "python", "cmake", "llvm", "curl"
			]

			missing = []
			for pkg in brew_packages:
				result = subprocess.run([BREW, "list", pkg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
				if result.returncode != 0:
					missing.append(pkg)

			if missing:
				log(f"Missing packages: {', '.join(missing)}\n")
				if messagebox.askyesno(
					"Install Missing Dependencies",
					f"The following packages are missing:\n{', '.join(missing)}\nInstall now?"
				):
					for pkg in missing:
						log(f"Installing {pkg}...\n")
						run_command(f"{BREW} install {pkg}")
				else:
					log("Dependencies not installed. Compile may fail.\n")
			else:
				log("All brew dependencies are satisfied.\n")

			# Rust
			env = os.environ.copy()
			cargo_bin = os.path.expanduser("~/.cargo/bin")
			env["PATH"] = f"{cargo_bin}:{env.get('PATH','')}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

			result = subprocess.run(["rustc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
			if result.returncode != 0:
				log("Rust not installed. Installing via rustup...\n")
				run_command("curl https://sh.rustup.rs -sSf | sh -s -- -y", env=env)
				run_command("rustup default stable", env=env)
				log("Rust installed.\n")
			else:
				log(f"Rust found: {result.stdout.strip()}\n")

			result = subprocess.run(["cargo", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
			if result.returncode != 0:
				log("Cargo not found after Rust install.\n")
			else:
				log(f"Cargo found: {result.stdout.strip()}\n")

			# LLVM for Electrs
			llvm_prefix = "/opt/homebrew/opt/llvm"
			if os.path.isdir(llvm_prefix):
				log(f"LLVM found at {llvm_prefix}\n")
				env["PATH"] = f"{llvm_prefix}/bin:" + env["PATH"]
				env["LIBCLANG_PATH"] = f"{llvm_prefix}/lib"
				env["DYLD_LIBRARY_PATH"] = f"{llvm_prefix}/lib"
			else:
				log("LLVM not found. Installing via Homebrew...\n")
				run_command(f"{BREW} install llvm")
				env["PATH"] = f"{llvm_prefix}/bin:" + env["PATH"]
				env["LIBCLANG_PATH"] = f"{llvm_prefix}/lib"
				env["DYLD_LIBRARY_PATH"] = f"{llvm_prefix}/lib"

			messagebox.showinfo("Dependency Check", "All dependencies for Bitcoin and Electrs are installed and ready!")

		except Exception as e:
			log(f"Error during dependency check: {e}\n")

	threading.Thread(target=task, daemon=True).start()

# ================== BUILD FUNCTIONS ==================
def compile_bitcoin_source(src, cores, needs_cmake):
	set_progress(0)
	binary_path = None

	if needs_cmake:
		build = os.path.join(src, "build")
		os.makedirs(build, exist_ok=True)
		run_command(
			"cmake -S . -B build "
			"-DBUILD_GUI=OFF "
			"-DBUILD_WALLET=OFF "
			"-DENABLE_TESTS=OFF "
			"-DENABLE_BENCH=OFF",
			cwd=src
		)
		set_progress(35)

		def progress_cb(line):
			if "Building" in line and progress_var.get() < 90:
				set_progress(progress_var.get() + 0.04)

		run_command(f"cmake --build build -j{cores}", cwd=src, progress_cb=progress_cb)

		for root_dir, dirs, files in os.walk(build):
			if "bitcoind" in files:
				binary_path = os.path.join(root_dir, "bitcoind")
				break

	else:
		run_command("./autogen.sh", cwd=src)
		set_progress(30)
		run_command("./configure --without-gui --disable-wallet", cwd=src)
		set_progress(35)

		def progress_cb(line):
			if ("CXX" in line or "CC" in line) and progress_var.get() < 90:
				set_progress(progress_var.get() + 0.04)

		run_command(f"make -j{cores}", cwd=src, progress_cb=progress_cb)
		possible_paths = [os.path.join(src, "src", "bitcoind")]
		for path in possible_paths:
			if os.path.isfile(path):
				binary_path = path
				break

	if binary_path and os.path.isfile(binary_path):
		log(f"bitcoind binary found at: {binary_path}\n")
		return binary_path
	else:
		log(f"Warning: bitcoind binary not found after build.\n")
		return None

def compile_electrs_source(src, cores):
	set_progress(0)
	env = os.environ.copy()
	cargo_bin = os.path.expanduser("~/.cargo/bin")
	env["PATH"] = f"{cargo_bin}:{env.get('PATH','')}:/opt/homebrew/bin:/usr/local/bin"
	llvm_prefix = "/opt/homebrew/opt/llvm"
	if os.path.isdir(llvm_prefix):
		env["PATH"] = f"{llvm_prefix}/bin:" + env["PATH"]
		env["LIBCLANG_PATH"] = f"{llvm_prefix}/lib"
		env["DYLD_LIBRARY_PATH"] = f"{llvm_prefix}/lib"

	run_command(f"cargo build --release -j {cores}", cwd=src, env=env)
	return os.path.join(src, "target", "release", "electrs")

# ================== MAIN COMPILE ==================
def compile_selected():
	try:
		compile_btn.config(state="disabled")
		log_text.delete("1.0", "end")
		set_progress(0)

		target = target_var.get()
		cores = int(core_var.get())
		base = build_dir_var.get()
		binaries_only = binaries_only_var.get()
		os.makedirs(base, exist_ok=True)

		# Helper to clean build folder but keep binaries
		def cleanup_build_folder(build_folder, binaries):
			for root, dirs, files in os.walk(build_folder):
				for f in files:
					full_path = os.path.join(root, f)
					if full_path not in binaries:
						try:
							os.remove(full_path)
						except:
							pass
				for d in dirs:
					full_path = os.path.join(root, d)
					try:
						shutil.rmtree(full_path)
					except:
						pass

		# --- Bitcoin ---
		if target in ("Bitcoin", "Both"):
			try:
				bitcoin_version = bitcoin_version_var.get()
				tarball = os.path.join(base, f"{bitcoin_version}.tar.gz")
				url = f"https://github.com/bitcoin/bitcoin/archive/refs/tags/{bitcoin_version}.tar.gz"
				set_progress(2)
				log(f"Downloading Bitcoin Core {bitcoin_version}...\n")
				run_command(f"curl -L -o '{tarball}' {url}")
				set_progress(5)
				run_command(f"tar -xzf '{tarball}' -C '{base}'")
				set_progress(8)
				bitcoin_src = os.path.join(base, f"bitcoin-{bitcoin_version.lstrip('v')}")
				binary = compile_bitcoin_source(bitcoin_src, cores, use_cmake(bitcoin_version))
				if binary:
					log(f"Bitcoin Core binary created at: {binary}\n")
				else:
					log("Bitcoin binary not found; skipping.\n")
				if binaries_only:
					cleanup_build_folder(bitcoin_src, [binary] if binary else [])
					if os.path.exists(tarball):
						os.remove(tarball)
				set_progress(100)
			except Exception as e:
				log(f"Error compiling Bitcoin: {e}\n")

		# --- Electrs ---
		if target in ("Electrs", "Both"):
			try:
				electrs_version = electrs_version_var.get()
				tarball = os.path.join(base, f"electrs-{electrs_version}.tar.gz")
				url = f"https://github.com/romanz/electrs/archive/refs/tags/{electrs_version}.tar.gz"
				set_progress(2)
				log(f"Downloading Electrs {electrs_version}...\n")
				run_command(f"curl -L -o '{tarball}' {url}")
				set_progress(5)
				run_command(f"tar -xzf '{tarball}' -C '{base}'")
				set_progress(8)
				electrs_src = os.path.join(base, f"electrs-{electrs_version.lstrip('v')}")
				binary = compile_electrs_source(electrs_src, cores)
				if binary:
					log(f"Electrs binary created at: {binary}\n")
				if binaries_only:
					cleanup_build_folder(electrs_src, [binary] if binary else [])
					if os.path.exists(tarball):
						os.remove(tarball)
				set_progress(100)
			except Exception as e:
				log(f"Error compiling Electrs: {e}\n")

		messagebox.showinfo("Done", "Selected builds completed successfully!")

	except Exception as e:
		messagebox.showerror("Error", str(e))
	finally:
		compile_btn.config(state="normal")

# ================== VERSION REFRESH ==================
def refresh_bitcoin_versions():
	def task():
		try:
			log("Checking for new Bitcoin Core releases...\n")
			r = requests.get(BITCOIN_API, timeout=10)
			r.raise_for_status()
			new_versions = []
			for rel in r.json():
				tag = rel["tag_name"]
				if "rc" in tag.lower():
					continue
				new_versions.append(tag)
				if len(new_versions) == 10:
					break
			if new_versions:
				bitcoin_version_var.set(new_versions[0])
				bitcoin_version_combo['values'] = new_versions
				log(f"Found {len(new_versions)} Bitcoin Core releases.\n")
			else:
				log("No new Bitcoin Core releases found.\n")
		except Exception as e:
			log(f"Error checking Bitcoin releases: {e}\n")
	threading.Thread(target=task, daemon=True).start()

def refresh_electrs_versions():
	def task():
		try:
			log("Checking for new Electrs releases...\n")
			r = requests.get(ELECTRS_API, timeout=10)
			r.raise_for_status()
			new_versions = []
			for rel in r.json():
				tag = rel["tag_name"]
				if "rc" in tag.lower():
					continue
				new_versions.append(tag)
				if len(new_versions) == 10:
					break
			if new_versions:
				electrs_version_var.set(new_versions[0])
				electrs_version_combo['values'] = new_versions
				log(f"Found {len(new_versions)} Electrs releases.\n")
			else:
				log("No new Electrs releases found.\n")
		except Exception as e:
			log(f"Error checking Electrs releases: {e}\n")
	threading.Thread(target=task, daemon=True).start()

# ================== GUI ==================
if not BREW:
	tk.Tk().withdraw()
	messagebox.showerror("Homebrew not found", "Homebrew was not found.\n\nPlease install it first from:\nhttps://brew.sh")
	sys.exit(1)

root = tk.Tk()
root.title("Bitcoin & Electrs Compiler")
root.geometry("780x720")

# --- Dependency Checker ---
ttk.Button(root, text="Check Dependencies", command=check_dependencies).pack(pady=5)

# --- Target selector ---
ttk.Label(root, text="Select Target to Compile").pack(pady=5)
target_var = tk.StringVar(value="Bitcoin")
ttk.Combobox(root, values=["Bitcoin", "Electrs", "Both"], textvariable=target_var, state="readonly", width=10).pack()

# --- Bitcoin version selector ---
bitcoin_version_frame = ttk.Frame(root)
bitcoin_version_frame.pack(pady=5)
ttk.Label(bitcoin_version_frame, text="Bitcoin Version").pack(side="left")
bitcoin_version_var = tk.StringVar()
bitcoin_versions = get_bitcoin_versions()
bitcoin_version_var.set(bitcoin_versions[0])
bitcoin_version_combo = ttk.Combobox(bitcoin_version_frame, values=bitcoin_versions, textvariable=bitcoin_version_var, state="readonly", width=25)
bitcoin_version_combo.pack(side="left", padx=5)
ttk.Button(bitcoin_version_frame, text="Check for new releases", command=refresh_bitcoin_versions).pack(side="left", padx=5)

# --- Electrs version selector ---
electrs_version_frame = ttk.Frame(root)
electrs_version_frame.pack(pady=5)
ttk.Label(electrs_version_frame, text="Electrs Version").pack(side="left")
electrs_version_var = tk.StringVar()
electrs_versions = get_electrs_versions()
electrs_version_var.set(electrs_versions[0])
electrs_version_combo = ttk.Combobox(electrs_version_frame, values=electrs_versions, textvariable=electrs_version_var, state="readonly", width=25)
electrs_version_combo.pack(side="left", padx=5)
ttk.Button(electrs_version_frame, text="Check for new releases", command=refresh_electrs_versions).pack(side="left", padx=5)

# --- CPU cores ---
max_cores = multiprocessing.cpu_count()
core_var = tk.StringVar(value=str(max_cores))
ttk.Label(root, text="CPU Cores").pack(pady=(10, 0))
ttk.Combobox(root, values=list(range(1, max_cores + 1)), textvariable=core_var, state="readonly", width=6).pack()

# --- Build directory ---
build_dir_var = tk.StringVar(value=DEFAULT_BUILD_DIR)
ttk.Label(root, text="Build Location").pack(pady=(10, 0))
dir_frame = ttk.Frame(root)
dir_frame.pack()
ttk.Entry(dir_frame, textvariable=build_dir_var, width=50).pack(side="left")
ttk.Button(dir_frame, text="Choose…", command=lambda: build_dir_var.set(filedialog.askdirectory(initialdir=build_dir_var.get()) or build_dir_var.get())).pack(side="left", padx=5)

# --- Advanced ---
binaries_only_var = tk.BooleanVar()
advanced = ttk.LabelFrame(root, text="Advanced")
advanced.pack(fill="x", padx=15, pady=10)
ttk.Checkbutton(advanced, text="Keep only final binaries (delete build files)", variable=binaries_only_var).pack(anchor="w", padx=10)

# --- Progress bar ---
progress_var = tk.DoubleVar()
progress = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress.pack(fill="x", padx=15, pady=10)

# --- Log terminal ---
log_text = tk.Text(root, height=20, wrap="none")
log_text.pack(fill="both", expand=True, padx=15, pady=5)
scrollbar = ttk.Scrollbar(log_text, command=log_text.yview)
log_text.config(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")

# --- Compile button ---
compile_btn = ttk.Button(root, text="Compile Selected", command=lambda: threading.Thread(target=compile_selected, daemon=True).start())
compile_btn.pack(pady=10)

root.mainloop()
