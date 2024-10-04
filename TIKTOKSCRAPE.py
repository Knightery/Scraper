import gspread
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import json

chrome_driver_path = "C:/Users/nyter/Downloads/chromedriver-win64/chromedriver.exe"
service = Service(chrome_driver_path)

print("Program started!")
gc = gspread.service_account(filename="googleauth.json")  # Add your own JSON, generated from Google.
sheet = gc.open_by_key('168PQUY1hL5gL6uvBbBVFIyeMN_Ej6-fMoc3UqbYsaN4').sheet1  # Put your own sheet here
options = Options()
#options.headless = True
options.add_experimental_option('excludeSwitches', ['enable-logging'])
browser = webdriver.Chrome(options=options)  # Remember to download and use your own webdriver.

print("Starting data collection...")

browser.get("https://tiktok.com/en")
count = 0

IDs = set({})
while count < 10000:
    try:
        browser.refresh()
        html = browser.execute_script("return document.documentElement.outerHTML;")
        soup = BeautifulSoup(html, "html.parser")
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        data = soup.find("script", {"id": "SIGI_STATE", "type": "application/json"})
        if data.text is not None:
            data = json.loads(data.text)
        for i in data["ItemModule"]:
            if i in IDs:
                print("duplicate")
                continue
            else:
                IDs.add(i)
            if count == 10000:
                break
            if count % 50 == 0 and count >= 50:
                time.sleep(60)
            count += 1
            print("Currently on Tiktok #" + str(count))

            # Using strings as bools
            # to stay consistent with
            # formatting

            stuff = {
                     "Description": data["ItemModule"][i]["desc"],
                     "Likes": data["ItemModule"][i]["stats"]["diggCount"],
                     "Shares": data["ItemModule"][i]["stats"]["shareCount"],
                     "Comments": data["ItemModule"][i]["stats"]["commentCount"],
                     "Plays": data["ItemModule"][i]["stats"]["playCount"],
                     }
            sheet.append_row([
                              stuff["Description"], int(stuff["Likes"]), int(stuff["Shares"]),
                              int(stuff["Comments"]), int(stuff["Plays"])
                              ])
            print("Description")
    except Exception as e:
        print(f"{e}")
        time.sleep(8)
count = 0

print("Program has finished.")