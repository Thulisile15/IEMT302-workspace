import argparse
import sys
from typing import List, Dict, Tuple, Optional

import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def detect_location_by_ip(timeout_seconds: int = 5) -> Optional[Tuple[float, float]]:
    """Attempt to detect approximate lat/lon from public IP.

    Returns None if detection fails.
    """
    try:
        response = requests.get(
            "https://ipinfo.io/json", headers={"User-Agent": USER_AGENT}, timeout=timeout_seconds
        )
        response.raise_for_status()
        data = response.json()
        loc = data.get("loc")
        if not loc:
            return None
        lat_str, lon_str = loc.split(",", 1)
        return float(lat_str), float(lon_str)
    except Exception:
        return None


def fetch_forecast_html(lat: float, lon: float, timeout_seconds: int = 10) -> str:
    """Fetch the NWS forecast HTML page for the provided coordinates.

    NWS coverage is U.S.-centric; outside the U.S. this may return an empty page.
    """
    url = f"https://forecast.weather.gov/MapClick.php?lat={lat:.4f}&lon={lon:.4f}&unit=0&lg=english&FcstType=dwml"
    # Use the default HTML view for parsing readable blocks
    url_html = f"https://forecast.weather.gov/MapClick.php?lat={lat:.4f}&lon={lon:.4f}"
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    response = requests.get(url_html, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def parse_forecast(html: str) -> Dict[str, object]:
    """Parse the NWS 7-day forecast HTML into a structured dict.

    Returns a dict with optional "location" and a list under "periods".
    """
    soup = BeautifulSoup(html, "html.parser")

    # Location title if available
    location_text: Optional[str] = None
    location_heading = soup.select_one("#seven-day-forecast, #seven-day-forecast-container")
    if location_heading:
        # Try common patterns for the location name
        title_candidate = location_heading.select_one("h2, .panel-title, .forecast-header")
        if title_candidate and title_candidate.get_text(strip=True):
            location_text = title_candidate.get_text(strip=True)

    periods: List[Dict[str, str]] = []
    for item in soup.select("#seven-day-forecast .tombstone-container"):
        period_name = item.find("p", class_="period-name")
        short_desc = item.find("p", class_="short-desc")
        temp = item.find("p", class_="temp")
        img = item.find("img")

        entry: Dict[str, str] = {}
        if period_name and period_name.get_text(strip=True):
            entry["period"] = period_name.get_text(strip=True)
        if short_desc and short_desc.get_text(strip=True):
            entry["short"] = short_desc.get_text(strip=True)
        if temp and temp.get_text(strip=True):
            entry["temp"] = temp.get_text(strip=True)
        if img and img.has_attr("title") and img["title"]:
            entry["detail"] = img["title"].strip()

        if entry:
            periods.append(entry)

    return {"location": location_text, "periods": periods}


def print_forecast(forecast: Dict[str, object]) -> None:
    location = forecast.get("location")
    if isinstance(location, str) and location:
        print(location)
        print("=" * len(location))
    else:
        print("7-Day Forecast")
        print("==============")

    periods = forecast.get("periods")
    if not isinstance(periods, list) or not periods:
        print("No forecast data found.")
        return

    for entry in periods:
        if not isinstance(entry, dict):
            continue
        period = entry.get("period", "?")
        temp = entry.get("temp", "")
        short = entry.get("short", "")
        detail = entry.get("detail", "")

        header_parts: List[str] = [period]
        if temp:
            header_parts.append(f"({temp})")
        header_line = " ".join(part for part in header_parts if part)
        print(f"- {header_line}")
        if short:
            print(f"  {short}")
        if detail and detail != short:
            print(f"  {detail}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Scrape local NWS 7-day forecast using Beautiful Soup"
    )
    parser.add_argument("--lat", type=float, help="Latitude, e.g. 37.7749")
    parser.add_argument("--lon", type=float, help="Longitude, e.g. -122.4194")
    args = parser.parse_args(argv)

    lat: Optional[float] = args.lat
    lon: Optional[float] = args.lon

    if lat is None or lon is None:
        auto = detect_location_by_ip()
        if auto is not None:
            lat, lon = auto
        else:
            # Default to San Francisco, CA if detection failed
            lat, lon = 37.7749, -122.4194

    try:
        html = fetch_forecast_html(lat, lon)
        forecast = parse_forecast(html)
        print_forecast(forecast)
        return 0
    except requests.HTTPError as http_err:
        print(f"HTTP error: {http_err}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
