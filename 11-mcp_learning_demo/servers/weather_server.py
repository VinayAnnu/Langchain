"""MCP server exposing a current-weather tool using Open-Meteo."""
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-server")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}


@mcp.tool()
def get_current_weather(city: str) -> str:
    """Get current weather conditions for a city, including temperature, humidity, and wind."""
    try:
        geo_response = requests.get(
            GEOCODING_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=15,
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        results = geo_data.get("results", [])

        if not results:
            return f"Could not find a location matching '{city}'. Try a more specific city name."

        place = results[0]
        latitude = place["latitude"]
        longitude = place["longitude"]

        weather_response = requests.get(
            WEATHER_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=15,
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        current = weather_data.get("current")

        if not current:
            return f"Weather data is currently unavailable for {city}."

        code = current.get("weather_code")
        description = WEATHER_CODES.get(code, f"Weather code {code}")
        location_name = ", ".join(
            part for part in [place.get("name"), place.get("admin1"), place.get("country")] if part
        )

        return (
            f"Current weather for {location_name} "
            f"(observed at {current.get('time', 'unknown time')}): "
            f"{description}; temperature {current.get('temperature_2m')} °C; "
            f"feels like {current.get('apparent_temperature')} °C; "
            f"humidity {current.get('relative_humidity_2m')}%; "
            f"wind {current.get('wind_speed_10m')} km/h; "
            f"precipitation {current.get('precipitation')} mm. "
            "Weather data provided by Open-Meteo."
        )
    except requests.RequestException as exc:
        return f"Unable to retrieve weather data right now: {exc}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
