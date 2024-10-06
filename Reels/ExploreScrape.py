from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

chrome_driver_path = "C:\\Users\\nyter\\Desktop\\Scraper\\chromedriver-win64\\chromedriver.exe"

# Set up the Chrome WebDriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("user-data-dir=C:/Users/nyter/AppData/Local/Google/Chrome/User Data")
chrome_options.add_argument('profile-directory=Profile 1')

service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("googleauth.json", scope)
client = gspread.authorize(creds)
sheet = client.open("MAIN").worksheet("ReelsData")

# Function to scroll and load more posts
def scroll_down(driver, num_scrolls):
    body = driver.find_element(By.TAG_NAME, 'body')
    for _ in range(num_scrolls):
        body.send_keys(Keys.PAGE_DOWN)
        body.send_keys(Keys.PAGE_DOWN)
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(2)  # Adjust delay based on your connection speed

# Function to append data in batches of 100
def append_to_sheet(data):
    for i in range(0, len(data), 100):
        batch = data[i:i+100]
        sheet.append_rows([[url] for url in batch])

try:
    while True:
        # Open Instagram Explore page
        driver.get('https://www.instagram.com/explore/')

        # Wait for the page to load
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

        # Scroll down multiple times to load more content
        scroll_down(driver, 10)

        # Extract all post links
        post_links = set()  # Using a set to avoid duplicates
        post_elements = driver.find_elements(By.TAG_NAME, 'a')

        for elem in post_elements:
            link = elem.get_attribute('href')
            if '/p/' in link:  # Filter only post links
                post_links.add(link)

        # Convert set to list and append to Google Sheet
        post_links_list = list(post_links)
        append_to_sheet(post_links_list)

        # Print the number of links found and added
        print(f"Found {len(post_links_list)} post links and added to Google Sheet.")

        # Wait before the next iteration
        time.sleep(20)  # Adjust the delay as needed

except KeyboardInterrupt:
    print("Script stopped by user.")

finally:
    # Close the driver
    driver.quit()
