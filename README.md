# BitForge Python is a GUI based Bitcoin & Electrs binary Compiler for macOS written in Python

A native macOS application that simplifies the process of compiling Bitcoin Core and Electrs from source. Built with Python and Tkinter, this app provides a user-friendly GUI to build production-ready Bitcoin node software on your Mac.

## Easy App Build Steps

### Homebrew
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### XXcode Command Line Tools
```
xcode-select --install
```

### Git via brew
```
brew install git
```

### Python via brew
```
brew install python
```

### Pyinstaller
```
pip3 install pyinstaller
```

### Build the BitForge App
```
git clone https://github.com/csd113/BitForge-Python.git && cd BitForge-Python && chmod +x build.sh && ./build.sh
```

## ✨ Features

- **One-Click Setup** - Automatically checks and installs all required dependencies via Homebrew
- **Visual Progress Tracking** - Real-time build logs and progress indicators
- **Version Selection** - Choose from the latest Bitcoin Core and Electrs releases
- **Multi-Core Compilation** - Utilize all your CPU cores for faster builds
- **Node-Optimized Builds** - Compiles Bitcoin Core without wallet dependencies (Berkeley DB not required)
- **Dual Architecture Support** - Works on both Apple Silicon (M1/M2/M3) and Intel Macs
- **PyInstaller Compatible** - Can be packaged as a standalone macOS app

## 🔨 What It Builds

### Bitcoin Core (Node-Only Build)
Compiles the following binaries without wallet support:

- **bitcoind** - The Bitcoin node daemon for running a full node
- **bitcoin-cli** - Command-line interface for interacting with bitcoind
- **bitcoin-tx** - Transaction manipulation tool
- **bitcoin-util** - Utility functions

> **Note:** This app builds Bitcoin Core in "node-only" mode (wallet disabled). This means:
> - ✅ Perfect for running a full Bitcoin node
> - ✅ No Berkeley DB dependency required
> - ✅ Faster compilation
> - ❌ Cannot create or manage wallets directly in bitcoind

### Electrs (Electrum Server)

Compiles the Electrs binary for running an Electrum server that connects to your Bitcoin node.

## 📦 Prerequisites

### System Requirements

- **macOS 10.15 (Catalina)** or later
- **Xcode Command Line Tools** installed
- **Homebrew** package manager
- **Python 3.8+** (usually pre-installed on macOS)

### Automatic Dependency Installation

The app can automatically install the following via Homebrew:

**Build Tools:**
- automake
- libtool
- pkg-config
- cmake

**Bitcoin Core Dependencies:**
- boost
- miniupnpc
- zeromq
- sqlite
- python

**Electrs Dependencies:**
- llvm
- rust (via rustup)

## 🚀 Installation

### Option 1: Package into MacOS app

1. **Clone the repository:**
   ```
   git clone https://github.com/csd113/bitcoin-and-electrs-compiler-macos.git
   cd bitcoin-and-electrs-compiler-macos
   ```

### 1. Make sure you have the files
Should see:
   - compile_bitcoind_gui.py
     
   - bitcoin_compiler.spec
     
   - build_app.sh

### 2. Run the build script
```
chmod +x build_app.sh

./build_app.sh
```
Done! App is in dist/ folder

The build script will:
- ✅ Check if PyInstaller is installed
- ✅ Install it if missing
- ✅ Build the app with correct settings
- ✅ Verify the build
- ✅ Test launch it
- ✅ Show next steps

### Build Requirements:
- macOS 10.13 or later
- Python 3.8 or later
- pip (Python package manager)

### Will be installed automatically:
- PyInstaller

### Build Method 2: Manual Command Line

```
pyinstaller \
    --name "Bitcoin Compiler" \
    --windowed \
    --onedir \
    --noconfirm \
    --clean \
    --osx-bundle-identifier com.bitcointools.compiler \
    compile_bitcoind_gui.py
```

### Option 2: Run without building app

1. **Clone the repository:**
   ```
   git clone https://github.com/csd113/bitcoin-and-electrs-compiler-macos.git
   cd bitcoin-and-electrs-compiler-macos
   ```

2. **Run the application:**
   ```bash
   python3 compile_bitcoind_gui.py
   ```

## 📖 Usage

### Step 1: Check Dependencies

1. Launch the application
2. Click **"Check & Install Dependencies"**
3. Review the list of installed/missing packages
4. Click **"Yes"** to install missing dependencies automatically

### Step 2: Configure Build

1. **Select Target:**
   - Bitcoin (only)
   - Electrs (only)
   - Both

2. **Choose CPU Cores:**
   - Default: Maximum cores - 1
   - Range: 1 to all available cores
   - More cores = faster compilation

3. **Set Build Directory:**
   - Default: `~/Downloads/bitcoin_builds`
   - Click "Browse" to change location

### Step 3: Select Versions

1. **Bitcoin Version:**
   - Auto-loads latest releases from GitHub
   - Click "Refresh" to update list
   - Select desired version (e.g., v28.0)

2. **Electrs Version:**
   - Auto-loads latest releases
   - Select desired version (e.g., v0.10.0)

### Step 4: Compile

1. Click **"🚀 Start Compilation"**
2. Monitor the build log for progress
3. Watch the progress bar
4. Wait for completion (this can take 15-60 minutes depending on your system)

## ⚙️ Build Options

### CMake Build (Bitcoin Core v28.0+)

The app uses modern CMake syntax:
```bash
cmake -B build -DENABLE_WALLET=OFF -DBUILD_GUI=OFF
cmake --build build -j<cores>
```

### Autotools Build (Bitcoin Core <v28.0)

For older versions:
```bash
./autogen.sh
./configure --disable-wallet --disable-gui
make -j<cores>
```

### Electrs Build

Uses Cargo (Rust's package manager):
```bash
cargo build --release --locked
```

## 📂 Output

Compiled binaries are saved to:
```
~/Downloads/bitcoin_builds/binaries/
├── bitcoin-<version>/
│   ├── bitcoind
│   ├── bitcoin-cli
│   ├── bitcoin-tx
│   ├── bitcoin-util
│   └── bitcoin-wallet
└── electrs-<version>/
    └── electrs
```

## 🔧 Troubleshooting

### "Homebrew not found"
Install Homebrew from [brew.sh](https://brew.sh):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### "Command failed: cmake"
Make sure you've run "Check & Install Dependencies" first. If issues persist, manually install cmake:
```bash
brew install cmake
```

### Rust/Cargo not found
The app will prompt you to install Rust. Run:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Then restart the app.

### Compilation takes forever
- Reduce the number of CPU cores in the settings
- Make sure you have at least 20GB of free disk space
- Close other resource-intensive applications

## ❓ FAQ

**Q: Do I need Berkeley DB?**  
A: No! This app builds Bitcoin Core without wallet support, so Berkeley DB is not required.

**Q: Can I use these binaries in production?**  
A: Yes, they are built from official Bitcoin Core and Electrs source code.

**Q: Will this work on Apple Silicon Macs?**  
A: Yes, fully compatible with M1, M2, and M3 Macs.

**Q: How much disk space do I need?**  
A: At least 20GB free for the build process (source code, build files, and binaries).

**Q: Can I compile specific versions?**  
A: Yes, select any version from the dropdown menus.

**Q: What if my version isn't in the list?**  
A: Click the "Refresh" button to fetch the latest releases from GitHub.

**Q: Where are the build logs stored?**  
A: Build logs are displayed in real-time in the app's terminal window.

**Q: Can I stop a compilation in progress?**  
A: Yes, close the app. Build files will remain in the build directory.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [Bitcoin Core](https://github.com/bitcoin/bitcoin) - The reference implementation of Bitcoin
- [Electrs](https://github.com/romanz/electrs) - Efficient Electrum Server in Rust
- [Homebrew](https://brew.sh) - The missing package manager for macOS
