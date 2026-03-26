# Change OpenBCI UI to Text Mode

This repository provides a concise, platform-aware checklist for switching OpenBCI-based user interfaces from graphical views to text output. Use these steps if you want console-friendly or log-friendly output instead of plots.

## Where OpenBCI GUI stores settings
- **Windows:** `C:\Users\<YourUsername>\AppData\Local\OpenBCI\`
- **macOS:** `~/Library/Application Support/OpenBCI/`
- **Linux:** `~/.config/OpenBCI/`

Common config filenames inside those folders:
- `config.json`
- `settings.json`
- `openbci_config.json`
- `gui_settings.ini`

## How to switch to text output
1. **Open the GUI preferences** and look for **Display / Output / View** options. Select the option closest to `Text`, `Console`, or `Data` view.
2. **If you can edit config files**, set keys like these to text-friendly values:
   - `display_format: "text"`
   - `output_mode: "text"`
   - `surface_format: "text"` or `ui_display: "text"`
   - Disable visual mode flags such as `visual_mode: false`
3. **Python or BrainFlow projects:** inspect your project directory for `config.json`, `settings.ini`, or similar. Adjust the same keys there.

## Quick-start script
A small helper script is provided in `tools/modify_openbci_config.py`. It searches common OpenBCI config locations and forces text-friendly settings.

```bash
python tools/modify_openbci_config.py       # updates any found configs
python tools/modify_openbci_config.py --dry-run  # shows what would change
python tools/modify_openbci_config.py --path /custom/path/config.json
```

The script makes a `.bak` copy before writing changes and skips files it cannot parse as JSON.

## What to check after changing settings
- Restart the OpenBCI GUI to ensure the new config is loaded.
- Confirm data now appears in text/console form instead of plots.
- Revert using the `.bak` file if you want to restore prior settings.

