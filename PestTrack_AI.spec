# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

base_dir = os.path.abspath(SPECPATH)
ctk_dir = os.path.join(base_dir, 'pest_env', 'Lib', 'site-packages', 'customtkinter')

added_datas = [
    (os.path.join(base_dir, 'models'), 'models'),
    (os.path.join(base_dir, 'sample_data'), 'sample_data'),
    (ctk_dir, 'customtkinter'),
]

hidden_imports = [
    'customtkinter',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'cv2',
    'torch',
    'torchvision',
    'ultralytics',
    'reportlab',
    'sqlite3',
    'matplotlib',
    'config',
    'gui.main_window',
    'ai_models.pipeline',
] + collect_submodules('customtkinter')

a = Analysis(
    ['app.py'],
    pathex=[base_dir],
    binaries=[],
    datas=added_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter.test', 'unittest', 'email.test'],
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
    name='PestTrack_AI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False, # GUI window only, no black terminal window
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
    upx=False,
    upx_exclude=[],
    name='PestTrack_AI',
)
