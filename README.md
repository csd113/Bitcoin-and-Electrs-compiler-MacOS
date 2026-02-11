# Bitcoin-and-Electrs-compiler-MacOS
python based app that checks dependancies and is able to auto compile both bitcoin binaries and elecrs binaries for MacOS


# How to install:
 1. Make sure you have the files
### Should see:
   compile_bitcoind_gui.py
   bitcoin_compiler.spec
   build_app.sh

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

### Method 2: Manual with Spec File

```
# Install PyInstaller if needed
pip3 install pyinstaller

# Build
pyinstaller bitcoin_compiler.spec
```

### Method 3: Manual Command Line

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
