from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

def fetch_world_population():
    """
    Fetch the latest world population using Selenium.
    """
    chrome_driver_path = "C:\\Users\\Karthik Chittoor\\Downloads\\chromedriver-win64\\chromedriver-win64"  # Replace with the actual path to ChromeDriver
    service = Service(chrome_driver_path)

    # Set up Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run Chrome in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    try:
        # Initialize the WebDriver
        driver = webdriver.Chrome(service=service, options=options)
        driver.get("https://www.worldometers.info/world-population/")
        
        # Locate the population element on the page
        population_element = driver.find_element(By.CLASS_NAME, "maincounter-number")
        if population_element:
            world_population = population_element.text.strip().replace(",", "")
            return world_population
        else:
            print("Failed to locate population data on the page.")
            return None
    finally:
        driver.quit()

if __name__ == "__main__":
    population = fetch_world_population()
    if population:
        print(f"World Population: {population}")
    else:
        print("Could not fetch world population data.")
