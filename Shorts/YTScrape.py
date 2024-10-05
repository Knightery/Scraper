from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
import random

chrome_driver_path = "C:/Users/nyter/Downloads/chromedriver-win64/chromedriver.exe"

# Set up ChromeOptions to use an existing profile
service = Service(chrome_driver_path)
options = webdriver.ChromeOptions()
#options.add_argument("--headless")  # Run in headless mode
#options.add_argument("--window-size=1920x1080")  # Set a virtual window size
#options.add_argument("--disable-gpu")  # Disabling GPU for headless mode stability
#options.add_argument("--no-sandbox")  # Required for certain environments (e.g., Docker)

# Initialize WebDriver with options and service
driver = webdriver.Chrome(service=service, options=options)

driver.get("https://www.youtube.com/shorts")

wait = WebDriverWait(driver, 10)
descriptions = []
unique_descriptions = set()
BATCH_SIZE = 30

# Wait for the page to load (adjust time as needed)
time.sleep(5)

while True:
    try:
        # Wait for a short time to allow for loading new content
        time.sleep(random.uniform(0.75, 0.85))

        # Get shorts elements without waiting for their presence
        shorts_elements = driver.find_elements(By.CSS_SELECTOR, 'h2.title.style-scope.reel-player-header-renderer')

        for short_element in shorts_elements:
            description = short_element.text.strip()
            if description and description not in unique_descriptions:  # Avoid duplicates
                descriptions.append(description)
                unique_descriptions.add(description)
                print(f"Collected: {description}")

                # Check if it's time to save the batch to Excel
                if len(descriptions) >= BATCH_SIZE:
                    # Read existing data from the Excel file (if it exists)
                    try:
                        existing_df = pd.read_excel("shorts_descriptions.xlsx")
                    except FileNotFoundError:
                        existing_df = pd.DataFrame(columns=['Description'])

                    # Create new DataFrame with the collected descriptions
                    new_df = pd.DataFrame(descriptions, columns=['Description'])

                    # Append the new descriptions to the existing DataFrame
                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)

                    # Save the updated DataFrame back to the Excel file
                    updated_df.to_excel("shorts_descriptions.xlsx", index=False)
                    print(f"Saved {len(descriptions)} descriptions to 'shorts_descriptions.xlsx'")

                    # Clear the descriptions after saving
                    descriptions.clear()
                    
        # Scroll down multiple times to load more shorts
        for _ in range(1):  # Scroll down 1 times
            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ARROW_DOWN)

    except Exception as e:
        print(f"An error occurred: {e}")
        break
    
# Close the current tab
driver.close()