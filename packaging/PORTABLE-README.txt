PixForge 0.1.0 - Stylized 3D Texture Painter
==============================================

HOW TO RUN
----------
1. Unzip anywhere you have write access (e.g. Documents\PixForge).
   Do not put it in Program Files - that folder is read-only for a normal user.
2. Double-click pixforge.exe.

This build does not need the Microsoft Visual C++ Redistributable. The runtime
is linked into the exe, so it should start on a clean machine with nothing else
installed.

THINGS TO KNOW
--------------
* A console window does not appear. To see the log, run it from a terminal:
      .\pixforge.exe
  Log level comes from RUST_LOG, e.g.  set RUST_LOG=debug

* First launch is slower while the wgpu pipeline compiles. That is normal.

* If the brush library looks empty, the brushes folder must sit in the same
  directory as pixforge.exe. To keep a separate library instead, point
  PIXFORGE_BRUSHES at it:
      set PIXFORGE_BRUSHES=D:\my-brushes
  Drop your own .png / .gbr files in there; subfolders become categories.

* Your UI layout and custom palettes are saved to %APPDATA%\pixforge, so
  upgrading is just replacing this folder's contents.

* Needs a GPU with a Vulkan, DX12 or Metal backend. There is no CPU fallback,
  so it will not run on a headless VM or a machine with no GPU drivers.

* Uninstaller: there isn't one in this portable build. Delete the folder.
  Settings in %APPDATA%\pixforge are left behind if you want to keep them.

FEEDBACK
--------
Please report what you hit: what you were doing, what you expected, and what
happened instead. Crash logs come from %LOCALAPPDATA%\Temp - set RUST_LOG and
attach the output if something fails at startup.

Licensed under the GNU GPL v3.0 or later; see LICENSE.txt.
