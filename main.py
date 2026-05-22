import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy as np
import os
import time
import requests

def extract_and_download(page_url, filename):
    """Extracts MP4 URL from NBA.com page and downloads it using browser session cookies.

    Args:
        page_url (str): The NBA.com event page URL.
        filename (str): The path to save the downloaded file.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--mute-audio')

    driver = webdriver.Chrome(options=options)
    driver.set_window_position(3000, 3000)
    driver.get(page_url)

    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'video')))

    visible_mp4_urls = [
        a.get_attribute('src') for a in driver.find_elements(By.TAG_NAME, 'video')
        if a.is_displayed() and a.get_attribute('src').endswith('.mp4')
    ]

    mp4_url = visible_mp4_urls[0]

    # Use requests with cookies from the live browser session
    session = requests.Session()
    for cookie in driver.get_cookies():
        session.cookies.set(cookie['name'], cookie['value'])

    headers = {
        'Referer': page_url,
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    response = session.get(mp4_url, headers=headers, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    driver.quit()


def main():
    # Path to your CSV
    csv_path = './data/sampled_data.csv'

    # Directory to save the downloaded files
    output_dir = './data/videos'
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path, sep=';')

    # Download each URL as an MP4
    force_download_all = False
    for counter, url in enumerate(df['urls'], start=1):
        try:
            filename = os.path.join(output_dir, f"clip_{counter}.mp4")

            # Skip if already downloaded
            if not force_download_all and os.path.exists(filename):
                print(f"Video clip {counter} already present")
                continue

            extract_and_download(url, filename)
            print(f"Successfully downloaded clip {counter}")

        except Exception as e:
            print(f"Failed to download {url} number {counter}: {e}")


if __name__ == "__main__":
    main()