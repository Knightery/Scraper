import re
from TikTokApi import TikTokApi
import asyncio
import os
import pandas as pd

token = input("MSTOKEN: ")
ms_token = os.environ.get(token, None)  # set your own ms_token

async def trending_videos():
    # Create an empty list to store the video data
    data = []

    async with TikTokApi() as api:
        await api.create_sessions(headless=False, ms_tokens=[ms_token], num_sessions=1, sleep_after=3)
        async for video in api.trending.videos(count=30):
            main_dict = video.as_dict

            # Get the description
            descriptions = find_desc(main_dict)

            # Get the statsV2 values
            stats = find_stats(main_dict)

            # Append each description and its associated stats to the data list
            for desc in descriptions:
                data.append([desc] + stats)

    # Convert the data into a DataFrame and write it to an Excel file
    df = pd.DataFrame(data, columns=["desc", "collectCount", "commentCount", "diggCount", "playCount", "repostCount", "shareCount"])
    excel_filename = "tiktok_data.xlsx"
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='TikTok Data')

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
    repostCount = statsV2.get('repostCount', 0)
    shareCount = statsV2.get('shareCount', 0)

    # Return the stats as a list
    return [collectCount, commentCount, diggCount, playCount, repostCount, shareCount]

if __name__ == "__main__":
    asyncio.run(trending_videos())
