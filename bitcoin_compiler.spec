# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Bitcoin & Electrs Compiler
Prevents double-launch issue on macOS

Usage:
    pyinstaller bitcoin_compiler.spec
"""

import sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

a = Analysis(
    ['compile_bitcoind_gui.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'tkinter.filedialog',
        'requests',
        'urllib3',
        'certifi',
        'charset_normalizer',
        'idna',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'PIL',
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BitcoinCompiler',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # CRITICAL: Must be False to prevent console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BitcoinCompiler',
)

app = BUNDLE(
    coll,
    name='Bitcoin Compiler.app',
    icon=None,  # Add your .icns file path here if you have one
    bundle_identifier='com.bitcointools.compiler',
    version='1.0.0',
    info_plist={
        'CFBundleName': 'Bitcoin Compiler',
        'CFBundleDisplayName': 'Bitcoin & Electrs Compiler',
        'CFBundleGetInfoString': 'Compile Bitcoin Core and Electrs from source',
        'CFBundleIdentifier': 'com.bitcointools.compiler',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHumanReadableCopyright': 'Copyright © 2024',
        'NSHighResolutionCapable': 'True',
        'LSMinimumSystemVersion': '10.13.0',
        'NSRequiresAquaSystemAppearance': 'False',
        # CRITICAL: These prevent double-launch
        'LSUIElement': '0',  # Show in Dock
        'LSBackgroundOnly': '0',  # Not a background app
        'NSPrincipalClass': 'NSApplication',
    },
)
