from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
from datetime import datetime

def initialize_driver(chrome_driver_path):
    """
    Initialize the Selenium WebDriver with ChromeDriver.
    """
    service = Service(chrome_driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=service, options=options)

def fetch_world_population(driver):
    """
    Fetch the latest world population using the given WebDriver.
    """
    try:
        driver.get("https://www.worldometers.info/world-population/")
        time.sleep(1)  # Allow time for dynamic content to load
        population_element = driver.find_element(By.CLASS_NAME, "maincounter-number")
        if population_element:
            world_population = population_element.text.strip().replace(",", "")
            return world_population
        else:
            print("Failed to locate population data on the page.")
            return "retrieving data..."
    except Exception as e:
        print(f"Error fetching world population data: {e}")
        return "retrieving data..."

def update_vault_file(file_path, population):
    """
    Update the vault.txt file with the latest world population.
    """
    try:
        with open(file_path, "w") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"Current world population is {population} (updated at {timestamp})\n")
        print(f"[{timestamp}] Updated {file_path} with the latest world population: {population}")
    except Exception as e:
        print(f"Error updating {file_path}: {e}")

if __name__ == "__main__":
    chrome_driver_path = "C:\\Users\\Karthik Chittoor\\Documents\\projects\\ollama-RAG\\easy-local-rag\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe"
    vault_file_path = "../vault.txt"
    
    # Initialize the WebDriver once
    driver = initialize_driver(chrome_driver_path)
    
    try:
        while True:
            # Fetch the latest world population
            latest_population = fetch_world_population(driver)
            if latest_population != "retrieving data...":
                # Update the vault.txt file
                update_vault_file(vault_file_path, latest_population)
            else:
                print("Skipping update due to fetch error.")
            
            # Wait for 10 seconds before the next update
            time.sleep(10)
    finally:
        # Quit the driver when done
        driver.quit()
