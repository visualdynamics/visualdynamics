; Inno Setup script for Visual Dynamics.
;   iscc /DMyAppVersion=0.0.1 packaging\installer.iss
#define MyAppName "Visual Dynamics"
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

[Setup]
AppId={{7C3F6A1E-4B2D-4E9A-9C7B-2F1D5A8E6B04}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Brandon Zwink
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=..\dist
OutputBaseFilename=VisualDynamics-{#MyAppVersion}-windows-x64-setup
Compression=lzma2/max
SolidCompression=yes
; per-user by default: no administrator prompt, which is one less
; obstacle for someone evaluating it on a managed machine
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=icon.ico
WizardStyle=modern

[Files]
Source: "..\dist\VisualDynamics\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\VisualDynamics.exe"
Name: "{group}\{#MyAppName} Run Log"; Filename: "{app}\VisualDynamics.exe"; Parameters: "--run-log"; Comment: "The run log alone in the window"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\VisualDynamics.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Registry]
; .vdyn opens in Visual Dynamics
Root: HKA; Subkey: "Software\Classes\.vdyn"; ValueType: string; ValueName: ""; ValueData: "VisualDynamics.Project"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\VisualDynamics.Project"; ValueType: string; ValueName: ""; ValueData: "Visual Dynamics Project"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\VisualDynamics.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\VisualDynamics.exe,0"
Root: HKA; Subkey: "Software\Classes\VisualDynamics.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\VisualDynamics.exe"" ""%1"""

[Run]
Filename: "{app}\VisualDynamics.exe"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
