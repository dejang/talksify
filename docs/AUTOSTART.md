# Auto-start on Login

## macOS — LaunchAgent

Create `~/Library/LaunchAgents/com.talksify.app.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.talksify.app</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Applications/Talksify.app/Contents/MacOS/talksify</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
```

Load it:
```bash
launchctl load ~/Library/LaunchAgents/com.talksify.app.plist
```

Unload it:
```bash
launchctl unload ~/Library/LaunchAgents/com.talksify.app.plist
```

Adjust the path if the app is installed elsewhere.

## Linux — Desktop Entry

Create `~/.config/autostart/talksify.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=Talksify
Exec=/usr/local/bin/talksify
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
Comment=Local voice dictation
```

Adjust `Exec` to match the installed binary path.

## Verifying

After logging out and back in, the tray icon should appear.
Press the configured hotkey (default: `Cmd+Space`) to confirm dictation works.
