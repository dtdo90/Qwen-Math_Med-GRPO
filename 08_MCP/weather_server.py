import requests
import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

ACCUWEATHER_API_KEY=os.getenv("ACCUWEATHER_API_KEY")
BASE_URL = "http://dataservice.accuweather.com"

# initialize server, specify port number
mcp=FastMCP("weather_server",port=8001)

@mcp.tool()
def get_current_weather(location: str):
    """ Get current weather information for a given location
        Args:
            location: input location, e.g. "Singapore"
        Returns:
            weather information for the location
    """
    try:
        # 1. Get location key
        location_response=requests.get(
            f"{BASE_URL}/locations/v1/cities/search",
            params={
                "apikey": ACCUWEATHER_API_KEY,
                "q": location,
                "language": "en-us",
                "details": "false"
            }
        )
        location_response.raise_for_status()
        locations=location_response.json()
        
        # If location not found, try with country code
        if not locations and "," not in location:
            location_with_country = f"{location}, SG"
            location_response = requests.get(
                f"{BASE_URL}/locations/v1/cities/search",
                params={
                    "apikey": ACCUWEATHER_API_KEY,
                    "q": location_with_country,
                    "language": "en-us",
                    "details": "false"
                }
            )
            location_response.raise_for_status()
            locations = location_response.json()
            
        if not locations:
            return {"error": f"Location '{location}' not found. Try using format 'City, Country' (e.g., 'Singapore, SG')"}
            
        location_key = locations[0]["Key"]
        
        # 2. Get current weather
        weather_response=requests.get(
            f"{BASE_URL}/currentconditions/v1/{location_key}",
            params={
                "apikey": ACCUWEATHER_API_KEY,
                "details": "true",
                "language": "en-us"
            }
        )
        weather_response.raise_for_status()
        weather_data=weather_response.json()
        if not weather_data:
            return {"error": "Weather data is not available"}
            
        # Extract only key information
        current = weather_data[0]
        return {
            "date_time": current['LocalObservationDateTime'],
            "temperature": f"{current['Temperature']['Metric']['Value']}°C",
            "condition": current['WeatherText'],
            "is_daytime": "Yes" if current['IsDayTime'] else "No"
        }
    
    except Exception as e:
        return {"error": str(e)}
    

if __name__=="__main__":
    mcp.run(transport="stdio")
    # print(get_current_weather(location="Singapore"))

