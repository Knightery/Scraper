import requests
from bs4 import BeautifulSoup
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from concurrent.futures import ThreadPoolExecutor
import time
from requests.exceptions import RequestException
from gspread.exceptions import APIError

# Authenticate and connect to Google Sheets
def connect_to_google_sheets(sheet_id, sheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    
    creds = ServiceAccountCredentials.from_json_keyfile_name('googleauth.json', scope)
    client = gspread.authorize(creds)
    
    # Open the Google Sheet
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.worksheet(sheet_name)
    return worksheet

# Get the Instagram description
def get_instagram_description(session, url):
    try:
        response = session.get(url, timeout=5)  # Set timeout to avoid long delays
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            meta_tag = soup.find('meta', property='og:description')
            if meta_tag:
                return meta_tag.get('content')
        return "Description not found"
    except Exception as e:
        return f"Error: {str(e)}"

# Find the first empty row efficiently by checking in chunks
def find_first_empty_row(worksheet, description_column, chunk_size=100):
    row = 2940
    while True:
        cell_range = worksheet.range(f'{description_column}{row}:{description_column}{row+chunk_size-1}')
        for i, cell in enumerate(cell_range):
            if not cell.value:
                return row + i
        row += chunk_size
        
# Retry function for batch updates
def retry_batch_update(worksheet, updates, retries=3, delay=2):
    for i in range(retries):
        try:
            worksheet.batch_update(updates)
            return True
        except APIError as e:
            print(f"APIError: {e}, Retrying in {delay} seconds...")
            time.sleep(delay)  # Wait before retrying
    print("Max retries reached. Skipping this batch.")
    return False

# Batch process Google Sheets with parallel requests
def process_reel_links_google_sheets(sheet_id, sheet_name, batch_size=100, max_workers=5):
    worksheet = connect_to_google_sheets(sheet_id, sheet_name)

    # Efficiently find the first empty row in the 'Description' column (assuming it's column B)
    first_empty_row = find_first_empty_row(worksheet, 'B')

    # Get the range of URLs starting from the first empty row
    urls = worksheet.col_values(1)[first_empty_row - 1:]  # Assuming URLs are in the first column

    updates = []
    futures = []
    
    # Use a session to optimize HTTP requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor, requests.Session() as session:
        for i, url in enumerate(urls, start=first_empty_row):
            url = url.strip()
            futures.append(executor.submit(get_instagram_description, session, url))

        # Collect results from futures
        for i, future in enumerate(futures, start=first_empty_row):
            description = future.result()
            print(f'Processing URL {i}: {description}')
            
            # Step 2: Clean the description to a single line
            cleaned_description = clean_description(description)

            # Step 3: Write the cleaned description back to the "Data" column (assumed to be column B)
            updates.append({
                'range': f'B{i}',  # Write cleaned description back to column B
                'values': [[cleaned_description]]
            })

            # Step 4: Insert extraction formula (e.g., description, animal, category, likes, comments, date)
            formula = f'=MID(B{i}, SEARCH(": """, B{i}) + 3, LEN(B{i}) - SEARCH(": """, B{i}) - 3)'
            updates.append({
                'range': f'D{i}',  # Assuming column D is where the extracted data is placed
                'values': [[formula]]
            })

            # Step 5: Apply Google Translate formula to translate the final description (column I to J)
            translate_formula = f'=GOOGLETRANSLATE(I{i},"auto","en")'
            updates.append({
                'range': f'J{i}',  # Assuming column J is for the translated description
                'values': [[translate_formula]]
            })

            # Batch update to Google Sheets with retry mechanism
            if len(updates) == batch_size:
                success = retry_batch_update(worksheet, updates, retries=5, delay=5)
                if success:
                    updates = []
                else:
                    print(f"Batch failed after retries. Skipping.")
                time.sleep(2)  # Adjust the delay between batch updates

    # Final batch update for remaining entries
    if updates:
        retry_batch_update(worksheet, updates, retries=5, delay=5)


# Example usage
process_reel_links_google_sheets('1HagDowZAKw5GM3cSn8A30clZBNOIwNq6Tz7Zsyh1O7c', 'ReelsData', batch_size=100, max_workers=5)
