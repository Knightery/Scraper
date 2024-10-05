import re
from TikTokApi import TikTokApi
import asyncio
import os
import pandas as pd
import time

token = input("MSTOKEN: ")
ms_token = os.environ.get(token, None)  # set your own ms_token

async def trending_videos():
    # Create an empty list to store the video data
    data = []
    num_videos = 30  # Number of videos to retrieve per batch
    total_videos = 675  # Total number of videos you want to retrieve

    async with TikTokApi() as api:
        await api.create_sessions(headless=False, ms_tokens=[ms_token], num_sessions=1, sleep_after=3)
        
        for _ in range(total_videos // num_videos):
            async for video in api.trending.videos(count=num_videos):
                main_dict = video.as_dict

                # Get the description
                descriptions = find_desc(main_dict)

                # Get the statsV2 values
                stats = find_stats(main_dict)

                # Append each description and its associated stats to the data list
                for desc in descriptions:
                    data.append([desc] + stats)

    # Convert the data into a DataFrame
    df = pd.DataFrame(data, columns=["desc", "collectCount", "commentCount", "diggCount", "playCount", "shareCount"])
    excel_filename = "tiktok_data.xlsx"

    # Read existing data from the Excel file (if it exists)
    try:
        existing_df = pd.read_excel(excel_filename)
    except FileNotFoundError:
        existing_df = pd.DataFrame(columns=["desc", "collectCount", "commentCount", "diggCount", "playCount", "shareCount"])

    # Append the new data to the existing DataFrame
    updated_df = pd.concat([existing_df, df], ignore_index=True)

    # Save the updated DataFrame back to the Excel file
    updated_df.to_excel(excel_filename, index=False)

def find_desc(given: dict) -> list:
    descriptions = []
    
    # Extracting descriptions
    def search_desc(d):
        if isinstance(d, dict):
            for key, value in d.items():
                if key == 'desc':
                    # Clean the description by removing the substring '', 
                    cleaned_value = value.replace(" '', ", "")
                    descriptions.append(cleaned_value)
    
    search_desc(given)
    return descriptions

def find_stats(given: dict) -> list:
    statsV2 = given.get('statsV2', {})
    # Extract the values for each of the requested keys in statsV2, default to 0 if not found
    collectCount = statsV2.get('collectCount', 0)
    commentCount = statsV2.get('commentCount', 0)
    diggCount = statsV2.get('diggCount', 0)
    playCount = statsV2.get('playCount', 0)
    shareCount = statsV2.get('shareCount', 0)

    # Return the stats as a list
    return [collectCount, commentCount, diggCount, playCount, shareCount]

if __name__ == "__main__":
    while True:
        asyncio.run(trending_videos())

