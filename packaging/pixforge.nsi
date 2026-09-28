; PixForge - NSIS installer (setup.exe)
;
; A single-file installer that needs no admin rights, as an alternative to the
; MSI that dist produces. Both install the same payload: the exe plus a brushes/
; folder, which must land beside the exe because the app resolves it relative to
; the executable.
;
; Build it by staging the payload and running makensis -DAPP_VERSION=...; the
; exact commands are in RELEASE.md.
;
; STATUS: this script has never been compiled. It was written without a way to
; test it - `makensis` needs NSIS plus MinGW, neither of which was available -
; so treat it as unverified until someone builds it on Windows. The MSI in
; wix/ is the supported path; see RELEASE.md.

!ifndef APP_VERSION
  !define APP_VERSION "0.1.0"
!endif
!ifndef APP_PUBLISHER
  !define APP_PUBLISHER "PixForge"
!endif

!define APP_NAME         "PixForge"
!define APP_EXE          "pixforge.exe"
!define APP_PUBLISHER_URL "https://github.com/krubka1/pixforge"

; Staged payload, relative to this script:
;   staging\pixforge.exe
;   staging\brushes\...
!define STAGING "..\staging"
!define OUTDIR  "..\target\package"

; Per-user install: the app keeps its state in %APPDATA%\pixforge and needs no
; service, driver or machine-wide registry write, so demanding admin would buy
; nothing. Switch to "admin" for a machine-wide install under Program Files.
!define REQUEST_EXECUTION "user"
!if "${REQUEST_EXECUTION}" == "admin"
  !define SHCTX HKLM
!else
  !define SHCTX HKCU
!endif

; Fail loudly on a stale staging dir instead of shipping a setup.exe with no
; brushes in it.
!ifnotexist "${STAGING}\${APP_EXE}"
  !error "Missing ${STAGING}\${APP_EXE} - run the staging step in RELEASE.md first."
!endif

Name "${APP_NAME}"
OutFile "${OUTDIR}\${APP_NAME}-${APP_VERSION}-setup.exe"
Unicode true
InstallDir "$LOCALAPPDATA\Programs\${APP_NAME}"
InstallDirRegKey HKCU "Software\${APP_NAME}" "InstallDir"
RequestExecutionLevel ${REQUEST_EXECUTION}
SetCompressor /SOLID lzma
ShowInstDetails show
ShowUninstDetails show

VIProductVersion "${APP_VERSION}"
VIAddVersionKey "ProductName"     "${APP_NAME}"
VIAddVersionKey "FileDescription" "${APP_NAME} - Stylized 3D Texture Painter"
VIAddVersionKey "CompanyName"     "${APP_PUBLISHER}"
VIAddVersionKey "LegalCopyright"  "Licensed under the GNU GPL v3.0 or later."
VIAddVersionKey "FileVersion"     "${APP_VERSION}"
VIAddVersionKey "ProductVersion"  "${APP_VERSION}"
VIAddVersionKey "Comments"        "${APP_PUBLISHER_URL}"

!include "MUI2.nsh"
!include "FileFunc.nsh"

!define MUI_ABORTWARNING
!define MUI_ICON   "..\assets\icon.ico"
!define MUI_UNICON "..\assets\icon.ico"
!define MUI_FINISHPAGE_RUN "$INSTDIR\${APP_EXE}"
!define MUI_FINISHPAGE_RUN_TEXT "Launch ${APP_NAME}"

!insertmacro MUI_PAGE_LICENSE "..\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; Add/Remove Programs entry, plus the install-size estimate that Explorer shows
; in the Programs list. Defined before the sections that call it.
!macro WRITE_UNINSTALL_KEYS
  WriteRegStr SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayName"     "${APP_NAME}"
  WriteRegStr SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayVersion"  "${APP_VERSION}"
  WriteRegStr SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "Publisher"       "${APP_PUBLISHER}"
  WriteRegStr SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayIcon"     "$INSTDIR\${APP_EXE}"
  WriteRegStr SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "InstallLocation" "$INSTDIR"
  WriteRegStr SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegStr SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "URLInfoAbout"    "${APP_PUBLISHER_URL}"
  WriteRegStr SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "HelpLink"        "${APP_PUBLISHER_URL}"
  WriteRegDWORD SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "NoModify" 1
  WriteRegDWORD SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "NoRepair" 1
  ; Add/Remove Programs shows this as the app's "size on disk", in KB.
  ; GetSize pushes size / file count / dir count into $0 / $1 / $2.
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  WriteRegDWORD SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "EstimatedSize" $0
!macroend

Section "PixForge" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"
  File "${STAGING}\${APP_EXE}"

  ; Stock brush library, beside the exe rather than in AppData: it is part of
  ; the install payload, and users drop their own .png/.gbr brushes in there.
  SetOutPath "$INSTDIR\brushes"
  File /r "${STAGING}\brushes\*.*"
  SetOutPath "$INSTDIR"

  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\${APP_NAME}" "InstallDir" "$INSTDIR"
  !insertmacro WRITE_UNINSTALL_KEYS
SectionEnd

Section "Start Menu shortcut" SectionStartMenu
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortCut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
  CreateShortCut "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Desktop shortcut" SectionDesktop
  CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
SectionEnd

Section "Uninstall"
  Delete "$INSTDIR\${APP_EXE}"
  ; Removes the whole brush folder, so brushes a user added to it go too.
  ; They are not recoverable, but silently leaving a stale library behind after
  ; an uninstall is worse.
  RMDir /r /REBOOTOK "$INSTDIR\brushes"
  Delete /REBOOTOK "$INSTDIR\Uninstall.exe"
  RMDir /REBOOTOK "$INSTDIR"

  Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk"
  RMDir "$SMPROGRAMS\${APP_NAME}"
  Delete "$DESKTOP\${APP_NAME}.lnk"

  DeleteRegKey HKCU "Software\${APP_NAME}"
  DeleteRegKey SHCTX "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"

  ; Tell Explorer to re-read the shell state we just changed.
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, i 0, i 0)'
SectionEnd
