# Boston's Culinary Landscape

This project analyzes Boston’s restaurant landscape using Yelp review data to uncover relationships between price, popularity, customer ratings, and neighborhood trends. The goal is to identify growth opportunities and understand consumer preferences to guide restaurateurs, city planners, and food enthusiasts.

## Features

* Data cleaning and preprocessing of Yelp restaurant and review datasets
* Correlation analysis between price, ratings, and popularity
* Neighborhood-level exploration of cuisines and restaurant trends
* Growth opportunity identification using descriptive statistics and scatter plots
* Interactive visualizations to communicate findings clearly

## Dataset

* The analysis uses a dataset titled “Yelp Reviews in Boston, MA” compiled by the Boston Area Research Initiative in partnership with Northeastern and Harvard. It includes:
* YELP.Reviews.csv — user review data (ratings, comments)
* YELP.Restaurants.csv — restaurant-level attributes like name, price level, and ratings
* YELP.CT.csv — neighborhood-level review density and pricing averages

## Analysis Workflow

### 1. Data Preprocessing

* Converted Yelp price labels ($, $$, $$$, $$$$) to numeric levels (1–4)
* Removed rows with missing price or rating data
* Standardized cuisine names to merge duplicate categories
* Adjusted overrepresented “Boston” neighborhood labels to avoid skewed visualizations

### 2. Statistical Insights

#### Correlation Analysis

* Price vs Rating: Weak positive correlation (r = 0.13) — higher-priced restaurants tend to have slightly better ratings.
* Price vs Popularity: Moderate positive correlation (r = 0.35) — expensive restaurants often have more reviews.
* Rating vs Review Count: Mild positive correlation (r = 0.21) — highly reviewed restaurants tend to receive slightly higher ratings

#### Cuisine & Neighborhood Trends
* Pizza, seafood, and American dominate the most popular cuisines.
* Brookline and Allston stand out as dining hotspots, with strong review density relative to restaurant count.
* Afghan and African cuisines receive some of the highest ratings but remain underrepresented, highlighting growth opportunities.