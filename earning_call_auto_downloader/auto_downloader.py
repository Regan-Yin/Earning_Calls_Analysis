# ISOM3350 Final Project
# Audio Downloader for Earnings Call
# Assist Analyze Calls Python code
# Author: Regan Yin
# Date: 2025-05-11
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains

# Allow user to input exchange and ticker
exchange      = input("Please enter the exchange (e.g. nasdaq/nyse): ").strip().lower()
ticker        = input("Please enter the stock ticker (e.g. aapl): ").strip().lower()
# Custom fiscal year and quarter range
start_year    = int(input("Please enter the start fiscal year (e.g. 2018): ").strip())
start_quarter = int(input("Please enter the start quarter (1-4): ").strip())
end_year      = int(input("Please enter the end fiscal year (e.g. 2025): ").strip())
end_quarter   = int(input("Please enter the end quarter (1-4): ").strip())

# Create audio folder
if not os.path.exists("audio"):
    os.makedirs("audio")

# Chrome download configuration
chrome_options = Options()
# chrome_options.add_argument('--headless')  # Uncomment for headless mode if needed
prefs = {
    "download.default_directory": os.path.abspath("audio"),
    "download.prompt_for_download": False,
    "directory_upgrade": True,
    "safebrowsing.enabled": True
}
chrome_options.add_experimental_option("prefs", prefs)

# Launch browser
browser = webdriver.Chrome(options=chrome_options)
wait    = WebDriverWait(browser, 15)

# Iterate through years and quarters serially, ensure each download completes before next quarter
for year in range(start_year, end_year + 1):
    for q in range(1, 5):
        if (year == start_year and q < start_quarter) or \
           (year == end_year   and q > end_quarter):
            continue

        # **Keep your original link format**
        url = f"https://earningscall.biz/e/{exchange}/s/{ticker}/y/{year}/q/q{q}"
        print(f"Visiting: {url}")
        browser.get(url)

        # Extract meeting date
        try:
            date_elem = wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.call-date-container p.text-date")
            ))
            date_text   = date_elem.text.strip()  # e.g. "05/01/2019"
            mm, dd, yy  = date_text.split("/")
            meeting_date = f"{yy}{int(mm):02d}{int(dd):02d}"
        except Exception:
            print("Meeting date not found, skipping...")
            continue

        # Click download button
        try:
            content_right = wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.content-right")
            ))
            download_btn = content_right.find_element(
                By.CSS_SELECTOR, "button.btn.download"
            )
            ActionChains(browser).move_to_element(download_btn).click(download_btn).perform()
            print("Clicked download button...")

            # —— Improved "single file wait and retry" logic —— 
            existing = set(os.listdir("audio"))
            # SITE‐DEFAULT PREFIX: e.g. "NFLX Q1 2025"
            default_prefix = f"{ticker.upper()} Q{q} {year}"
            cr_name = None

            for attempt in range(3):
                # Wait a short time to see if .crdownload appears
                start_ts = time.time()
                while time.time() - start_ts < 20:
                    added = set(os.listdir("audio")) - existing
                    crs = [f for f in added if f.endswith(".crdownload")]
                    if crs:
                        cr_name = crs[0]
                        print(f"[DOWNLOAD] Detected download: {cr_name}")
                        break
                    time.sleep(1)

                if cr_name:
                    break

                # If .crdownload not detected, check if final .mp3 already exists
                finals = [f for f in os.listdir("audio")
                        if f.startswith(default_prefix) and f.endswith(".mp3")]
                if finals:
                    print(f"[FOUND] Site-default file exists: {finals[0]}")
                    break

                # Retry clicking download button
                print(f"[RETRY CLICK] No download found, retry {attempt+2}")
                ActionChains(browser).move_to_element(download_btn).click(download_btn).perform()
                time.sleep(2)

            # Check again: if neither crdownload nor site-default mp3, skip
            finals = [f for f in os.listdir("audio")
                    if f.startswith(default_prefix) and f.endswith(".mp3")]
            if not cr_name and not finals:
                print("[SKIP] Download never started; skipping this quarter.")
                continue

            # If site-default mp3 exists, use it directly
            if finals and not cr_name:
                final_name = finals[0]
                print(f"[SKIP WAIT] Using existing: {final_name}")
            else:
                # Wait for crdownload to disappear
                cr_path = os.path.join("audio", cr_name)
                while os.path.exists(cr_path):
                    time.sleep(1)
                final_name = cr_name[:-len(".crdownload")]
                print(f"[DONE] Download finished: {final_name}")

            # Rename to your desired format: "TICKER_YEAR_QX_DATE.mp3"
            ext       = os.path.splitext(final_name)[1]
            old_path  = os.path.join("audio", final_name)
            new_name  = f"{ticker.upper()}_{year}_Q{q}_{meeting_date}{ext}"
            os.rename(old_path, os.path.join("audio", new_name))
            print(f"Saved: {new_name}")

        except (TimeoutException, NoSuchElementException):
            print("Download button not found, skipping...")
            continue
        except Exception as e:
            print(f"Error during download or renaming: {e}")
            continue

# Close browser
browser.quit()
print("All download tasks completed.")