# -*- coding: utf-8 -*-
import datetime
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from lxml import html
from sklearn.ensemble import IsolationForest


class RightmoveData:
    """The `RightmoveData` web scraper collects structured data on properties
    returned by a search performed on www.rightmove.co.uk.
    """

    def __init__(self, url: str, get_floorplans: bool = False, get_json_data: bool = False):
        """Initialize the scraper with a URL from the results of a property search."""
        self._url = url
        self._status_code, self._first_page = self._request(url)
        self._validate_url()
        self._results = self._get_results(get_floorplans=get_floorplans, get_json_data=get_json_data)

    @staticmethod
    def _request(url: str):
        """Send GET request and return the status code and content."""
        return requests.get(url).status_code, requests.get(url).content

    def refresh_data(self, url: str = None, get_floorplans: bool = False, get_json_data: bool = False):
        """Make a fresh GET request to refresh the data."""
        url = url or self._url  # If no new URL, use the existing one
        self._status_code, self._first_page = self._request(url)
        self._url = url
        self._validate_url()
        self._results = self._get_results(get_floorplans, get_json_data)

    def _validate_url(self):
        """Validate that the URL is from Rightmove and returns a valid status code."""
        valid_protocols = ["http", "https"]
        valid_types = ["property-to-rent", "property-for-sale", "new-homes-for-sale"]
        valid_urls = [f"{protocol}://www.rightmove.co.uk/{type}/find.html?" for protocol in valid_protocols for type in
                      valid_types]

        if not any(self._url.startswith(url) for url in valid_urls) or self._status_code != 200:
            raise ValueError(f"Invalid Rightmove URL:\n\n\t{self._url}")

    @property
    def url(self):
        return self._url

    @property
    def get_results(self):
        """Return the results as a Pandas DataFrame."""
        return self._results

    @property
    def results_count(self):
        """Get the number of results."""
        return len(self.get_results)

    @property
    def average_price(self):
        """Get the average price of the results, ignoring null prices."""
        return self.get_results["price"].dropna().mean()

    def summary_stats(self, by: str = None):
        """Summary statistics of the results, grouped by a column (e.g., 'number_bedrooms').

        Includes count, mean, median, std, min, and max for the price column.
        """
        # Default to 'type' for commercial properties, 'number_bedrooms' for residential
        by = by or ("type" if "commercial" in self.rent_or_sale else "number_bedrooms")
        assert by in self.get_results.columns, f"Column not found in `get_results`: {by}"

        # Drop rows where 'price' is missing
        df = self.get_results.dropna(subset=["price"])

        # Group by the specified column and calculate various statistics for 'price'
        summary = df.groupby(by)["price"].agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max"
        ).reset_index()

        # Sort by number of bedrooms or count for commercial properties
        if "number_bedrooms" in summary.columns:
            summary["number_bedrooms"] = summary["number_bedrooms"].astype(int)
            summary.sort_values(by="number_bedrooms", inplace=True)
        else:
            summary.sort_values(by="count", ascending=False, inplace=True)

        return summary

    @property
    def rent_or_sale(self):
        """Return whether the search is for rent or sale properties."""
        if "/property-for-sale/" in self._url or "/new-homes-for-sale/" in self._url:
            return "sale"
        elif "/property-to-rent/" in self._url:
            return "rent"
        elif "/commercial-property-for-sale/" in self._url:
            return "sale-commercial"
        elif "/commercial-property-to-let/" in self._url:
            return "rent-commercial"
        else:
            raise ValueError(f"Invalid Rightmove URL:\n\n\t{self._url}")

    @property
    def results_count_display(self):
        """Get the number of listings displayed on the first page."""
        tree = html.fromstring(self._first_page)
        xpath = "//span[@class='searchHeader-resultCount']/text()"
        return int(tree.xpath(xpath)[0].replace(",", ""))

    @property
    def page_count(self):
        """Get the total number of pages of results."""
        pages = self.results_count_display // 24 + (1 if self.results_count_display % 24 else 0)
        return min(pages, 42)  # Limit to 42 pages

    def _get_page(self, request_content: str, get_floorplans: bool = False, get_json_data: bool = False):
        """Scrape a single page of results."""
        tree = html.fromstring(request_content)

        # Define xpaths based on property type
        xp_prices = "//span[@class='propertyCard-priceValue']/text()" if "rent" in self.rent_or_sale else "//div[@class='propertyCard-priceValue']/text()"
        xp_titles = "//div[@class='propertyCard-details']//a[@class='propertyCard-link']//h2[@class='propertyCard-title']/text()"
        xp_addresses = "//address[@class='propertyCard-address']//span/text()"
        xp_weblinks = "//div[@class='propertyCard-details']//a[@class='propertyCard-link']/@href"
        xp_agent_urls = "//div[@class='propertyCard-contactsItem']//div[@class='propertyCard-branchLogo']//a[@class='propertyCard-branchLogo-link']/@href"
        xp_json_model = "//script[starts-with(text(), 'window.jsonModel = ')]/text()"

        # Extract data using xpaths
        price_pcm = tree.xpath(xp_prices)
        titles = tree.xpath(xp_titles)
        addresses = tree.xpath(xp_addresses)
        weblinks = [f"http://www.rightmove.co.uk{link}" for link in tree.xpath(xp_weblinks)]
        agent_urls = [f"http://www.rightmove.co.uk{link}" for link in tree.xpath(xp_agent_urls)]
        json_model = tree.xpath(xp_json_model)

        # Optionally scrape floorplan links and JSON data
        floorplan_urls = self._get_floorplans(weblinks) if get_floorplans else []
        json_data = self._get_json_data(json_model) if get_json_data else []


        # Define columns dynamically
        columns = ["price", "type", "address", "url", "agent_url"]

        # Combine all data into a list
        data = [price_pcm, titles, addresses, weblinks, agent_urls]

        # Add 'floorplan_url' if get_floorplans is True
        if get_floorplans:
            data.append(floorplan_urls)
            columns.append("floorplan_url")

        # Add JSON-related columns if get_json_data is True
        if get_json_data:
            data += json_data
            columns += ["subtype", "floorsize_sqft", "summary", "location", "nearest_postcode"]

        # Create DataFrame after defining all data and column names
        df = pd.DataFrame(data).transpose()

        # Assign the column names to the DataFrame
        df.columns = columns

        return df[df["address"].notnull()]

    def _get_floorplans(self, weblinks: list):
        """Scrape floorplan links from individual property pages."""
        floorplan_urls = []
        for weblink in weblinks:
            status_code, content = self._request(weblink)
            if status_code == 200:
                tree = html.fromstring(content)
                xp_floorplan_url = "//*[@id='floorplanTabs']/div[2]/div[2]/img/@src"
                floorplan_urls.append(tree.xpath(xp_floorplan_url)[0] if tree.xpath(xp_floorplan_url) else np.nan)
        return floorplan_urls

    def _get_json_data(self, json_model: list):
        """Extract JSON data from the model."""
        if json_model:
            json_text = json_model[0].split('window.jsonModel =', 1)[1].strip().rstrip(';')
            json_object = json.loads(json_text)
            json_data = [
                [p['propertySubType'], p['displaySize'], p['summary'], p['location']] + [
                    self.find_nearest_postcode(p['location']['latitude'], p['location']['longitude'])
                ] for p in json_object['properties']
            ]
            # Transpose the data so each attribute is in its own list (columns)
            json_data = [list(row) for row in zip(*json_data)]  # Transpose the list of lists
            return json_data
        return []

    def _get_results(self, get_floorplans: bool = False, get_json_data: bool = False):
        """Scrape results from all pages and return them as a DataFrame."""
        results = self._get_page(self._first_page, get_floorplans, get_json_data)
        for p in range(1, self.page_count + 1):
            p_url = f"{self._url}&index={p * 24}"
            status_code, content = self._request(p_url)
            if status_code == 200:
                temp_df = self._get_page(content, get_floorplans, get_json_data)
                results = pd.concat([results, temp_df])
        return self._clean_results(results)

    @staticmethod
    def find_nearest_postcode(latitude, longitude):
        """Find the nearest UK postcode based on latitude and longitude."""
        url = "https://api.postcodes.io/postcodes"
        params = {'lat': latitude, 'lon': longitude}

        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == 200:
                result = data.get('result')
                if result:
                    return result[0].get('postcode')
                else:
                    # No postcode found; return None.
                    return None
            else:
                raise ValueError(f"Error in response: {data['error']}")

        except requests.exceptions.Timeout:
            return None

    @staticmethod
    def _clean_results(results: pd.DataFrame):
        """Clean and format the results DataFrame."""
        # Convert price to numeric, remove non-numeric characters
        results["price"] = pd.to_numeric(results["price"].replace(regex=True, to_replace=r"\D", value=""))

        # Extract postcode, bedroom count, and other useful data
        results["postcode"] = results["address"].str.extract(r"\b([A-Za-z][A-Za-z]?[0-9][0-9]?[A-Za-z]?)\b")
        results["full_postcode"] = results["address"].str.extract(
            r"([A-Za-z][A-Za-z]?[0-9][0-9]?[A-Za-z]?[0-9]?\s[0-9]?[A-Za-z][A-Za-z])")
        results["number_bedrooms"] = results["type"].str.extract(r"\b([\d][\d]?)\b").astype(float)
        results["type"] = results["type"].str.strip()

        # Add search date
        results["search_date"] = datetime.datetime.now()

        # Handle floorsize as numeric, calculate price per square foot
        if "floorsize_sqft" in results.columns:
            results["floorsize_sqft"] = pd.to_numeric(results["floorsize_sqft"].str.replace(r"\D", "", regex=True),
                                                      errors="coerce")
            results["price_per_sqft"] = results["price"] / results["floorsize_sqft"]

        # Return the cleaned DataFrame
        return results.reset_index(drop=True)

    def plot_price_distribution(self):
        """Plot the distribution of property prices using self.get_results."""
        df = self.get_results  # Use self.get_results as the DataFrame

        # Check if the 'price' column exists in the DataFrame
        if "price" not in df.columns:
            print("Error: 'price' column not found in the data.")
            return

        plt.figure(figsize=(10, 6))
        df['price'].hist(bins=20, edgecolor='black')
        plt.title('Price Distribution of Properties')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.show()

    def complete_stats(self):
        """Generates summary statistics and visualizations for the data stored in self.get_results."""
        df = self.get_results  # Use self.get_results as the DataFrame

        # 1. Descriptive Statistics
        summary_statistics = df.describe(include='all')
        print("Summary Statistics:")
        print(summary_statistics)

        # Including mode (most frequent value)
        mode_values = df.mode().iloc[0]
        print("\nMode values:")
        print(mode_values)

        # 2. Histograms for each numerical column
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols].hist(figsize=(10, 8), bins=20)
        plt.suptitle('Histograms')
        plt.show()

        # 3. Boxplots for each numerical column
        plt.figure(figsize=(10, 6))
        df[num_cols].boxplot()
        plt.title('Boxplots')
        plt.show()

        # 4. Scatter plots for relationships between numerical variables
        sns.pairplot(df[num_cols])
        plt.suptitle('Scatter Plots', y=1.02)
        plt.show()

        # 5. Correlation matrix and heatmap
        if len(num_cols) > 1:  # Only plot if there are at least 2 numerical columns
            correlation_matrix = df[num_cols].corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Matrix Heatmap')
            plt.show()

