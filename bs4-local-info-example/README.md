# Beautiful Soup: Local NWS 7‑Day Forecast Example

This example shows how to use Beautiful Soup to scrape interesting local information: the U.S. National Weather Service (NWS) 7‑day forecast for your approximate location (detected by IP) or for coordinates you provide.

> Note: NWS coverage is U.S.-centric. Outside the U.S., results may be empty.

## Quickstart

```bash
cd bs4-local-info-example
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option A: auto-detect approximate coordinates via your public IP
python nws_forecast.py

# Option B: provide coordinates explicitly (example: New York City)
python nws_forecast.py --lat 40.7128 --lon -74.0060
```

## Output
A concise 7‑day view with period names, temperatures, and details, e.g.:

```
Seven Day Forecast for San Francisco, CA
=======================================
- Tonight (Low: 54 °F)
  Mostly cloudy
  Increasing clouds late, with patchy fog.
- Saturday (High: 64 °F)
  Partly sunny
  Light winds; sunshine later.
...
```

## How it works
- Uses `requests` to fetch the NWS forecast page for a set of coordinates
- Parses HTML with `beautifulsoup4` selecting the `#seven-day-forecast` tombstone blocks
- Extracts period, short description, temperature, and detailed tooltip text

## Troubleshooting
- If auto-detection fails, the script falls back to San Francisco, CA
- Some locations may have minor HTML differences; the parser intentionally handles missing fields
