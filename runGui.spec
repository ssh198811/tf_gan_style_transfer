# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

data_map = [("ui/icons.tga","ui"),("tf_model_p","tf_model_p"),("icons","icons")]

a = Analysis(['runGui.py'],
             pathex=['E:\\Users\\shishaohua.SHISHAOHUA1\\Downloads\\gan_tf\\ganilla'],
             binaries=[],
             datas=data_map,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='runGui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False , icon='E:\\Users\\shishaohua.SHISHAOHUA1\\Downloads\\gan_tf\\ganilla\\main.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='runGui')