import pandas as pd
import os, time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.common.exceptions import UnexpectedAlertPresentException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import UnexpectedAlertPresentException, NoAlertPresentException, TimeoutException

try:
    from webdriver_manager.chrome import ChromeDriverManager
    _use_manager = True
except ImportError:
    _use_manager = False

error_log_file = "error_log.txt"
no_data = []

# Helper function to collect data for a single URL
def collect_data_for_url(driver, accession_id, platform):
    if accession_id in no_data:
        print(f"Ignoring {accession_id} as it has no data.")
        return None
    url = f"https://www.ncbi.nlm.nih.gov/geo/geo2r/?acc={accession_id}&platform={platform}" if platform else f"https://www.ncbi.nlm.nih.gov/geo/geo2r/?acc={accession_id}"
    print(url)
    driver.get(url)
    driver.implicitly_wait(10)
    
    try:
        WebDriverWait(driver, 10).until(EC.alert_is_present())
        alert = driver.switch_to.alert
        alert_text = alert.text
        alert.accept()
        with open(error_log_file, 'a') as log_file:
            log_file.write(f"Error for {accession_id} with platform {platform}: {alert_text}\n")
        return None
    except TimeoutException:
        pass
    
    html_content = driver.page_source
    soup = BeautifulSoup(html_content, 'html.parser')

    # Parse the table
    table_container = soup.find('div', class_='samplesTableContainer')
    if table_container:
        table = table_container.find('table')
        if table:
            rows = table.find_all('tr')
            headers = []
            data = []
            for i, row in enumerate(rows):
                if i == 0:
                    headers = [th.get_text(strip=True) for th in row.find_all('th')]
                else:
                    cells = [td.get_text(strip=True) for td in row.find_all('td')]
                    if cells:
                        data.append(cells)
            return pd.DataFrame(data, columns=headers)  # Return a DataFrame
    return None

# Locate the Chrome or Chromium binary across macOS, Linux, and Windows.
def _find_chrome_binary():
    import shutil, platform
    candidates = []
    system = platform.system()
    if system == "Darwin":
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    elif system == "Linux":
        candidates = [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/snap/bin/chromium",
        ]
    elif system == "Windows":
        candidates = [
            os.path.join(os.environ.get("PROGRAMFILES", "C:\\Program Files"), "Google\\Chrome\\Application\\chrome.exe"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)"), "Google\\Chrome\\Application\\chrome.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google\\Chrome\\Application\\chrome.exe"),
        ]
    
    # Also check PATH for chromium / google-chrome
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            candidates.insert(0, found)
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None  # Let Selenium try to find it on its own

def _find_chromedriver_binary():
    """Find a system-installed chromedriver binary (e.g. installed via apt in Docker)."""
    import shutil
    for name in ("chromedriver", "chromium-driver", "chromium.chromedriver"):
        found = shutil.which(name)
        if found:
            return found
    for path in ("/usr/bin/chromedriver", "/usr/lib/chromium/chromedriver",
                 "/usr/lib/chromium-browser/chromedriver"):
        if os.path.exists(path):
            return path
    return None

# Helper Function to create a Chrome WebDriver compatible with headless/Linux environments.
def _create_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    binary = _find_chrome_binary()
    if binary:
        print(f"Using Chrome binary: {binary}")
        options.binary_location = binary

    # Prefer system chromedriver (already matched to installed Chromium, e.g. in Docker)
    system_driver = _find_chromedriver_binary()
    if system_driver:
        print(f"Using system ChromeDriver: {system_driver}")
        return webdriver.Chrome(service=Service(system_driver), options=options)

    # Fall back to webdriver-manager (downloads matching driver — works on macOS/Windows)
    if _use_manager:
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    return webdriver.Chrome(options=options)

# Main data processing function
def process_geo_data(input_file):
    data = pd.read_excel(input_file, sheet_name='Sheet1')
    output_folder = 'case-1'
    os.makedirs(output_folder, exist_ok=True)

    no_platform_file = os.path.join(output_folder, 'no_platforms.txt')
    driver = _create_driver()  # Ensure ChromeDriver is installed and in PATH
    
    no_data_file = os.path.join(output_folder, 'no_data.txt')
    
    # get no platforms list if exists to make a list and avoid processing them again
    no_data = []
    if os.path.exists(no_platform_file):
        with open(no_platform_file, 'r') as np_file:
            no_data = [line.split(':')[0] for line in np_file.readlines()]
    if os.path.exists(no_data_file):
        with open(no_data_file, 'r') as nd_file:
            no_data += [line.split(':')[0] for line in nd_file.readlines()]

    n = 0   
    for _, row in data.iterrows():
        accession_id = row['SeriesAccession']
        platforms = str(row['Platform(s)']).strip()
        output_file = os.path.join(output_folder, f'samples_table_{accession_id}.csv')

        # Skip if data already collected or is in no-platforms.txt file
        if os.path.exists(output_file):
            #print(f"Already collected data for {accession_id}, skipping...")
            continue
        elif os.path.exists(f"case-1/samples_table_{accession_id}.csv"):
            #print(f"Data already collected for {accession_id}.")
            continue
        elif accession_id in no_data:
            #print(f"Ignoring {accession_id} as it has no data.")
            continue
        
        # Process platforms
        if platforms.lower() == 'nan' or not platforms:
            with open(no_platform_file, 'a') as np_file:
                np_file.write(f"{accession_id}: No platforms available.\n")
            continue

        platform_list = platforms.split()
        platform_dfs = []

        for platform in platform_list:
            platform_df = collect_data_for_url(driver, accession_id, platform)
            if platform_df is not None:
                # Prefix column headers with the platform name
                platform_df.columns = [f"{platform}_{col}" for col in platform_df.columns]
                platform_dfs.append(platform_df)

        if platform_dfs:
            # Combine all platform DataFrames horizontally (side by side)
            combined_df = pd.concat(platform_dfs, axis=1)
            combined_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"Data successfully saved to {output_file}")
        else:
            print(f"No data found for {accession_id} with platforms: {platforms}")
            # save to file
            with open(no_data_file, 'a') as nd_file:
                nd_file.write(f"{accession_id}: No data found for platforms: {platforms}\n")
            time.sleep(1)
            # wait to not spam

    driver.quit()

if __name__ == '__main__':
    input_file = 'adj-chemo_metadata_split_fixed_series_accession.xlsx'
    process_geo_data(input_file)