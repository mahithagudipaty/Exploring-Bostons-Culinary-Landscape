#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS2500 Programming With Data
Final Project: Boston Restaurant Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
'''
Sources used: 
 https://www.geeksforgeeks.org/setting-the-color-of-bars-in-a-seaborn-barplot/
'''

def load_data(filename):
    """
    Load data from a CSV file into a pandas DataFrame.
    Parameters: filename
    Returns: loaded dataset
    """
    data = pd.read_csv(filename)
    return data

def load_all_data():
    """
    Load all datasets needed for the project.
    Returns: dataframes with csv datasets 
    """
    reviews_file = 'YELP.Reviews.csv'
    restaurants_file = 'YELP.Restaurants.csv'
    neighborhoods_file = 'YELP.CT.csv'

    reviews = load_data(reviews_file)
    restaurants = load_data(restaurants_file)
    neighborhoods = load_data(neighborhoods_file)

    return reviews, restaurants, neighborhoods

def preprocess_data(df):
    """
    process the data to ensure  columns are properly formatted.
    Parameters: the dataset
    Returns: the cleaned dataset
    """
    price_mapping = {'$ ': 1, '$$ ': 2, '$$$ ': 3, '$$$$ ': 4}
    df['price'] = df['price'].map(price_mapping)

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    df['restaurant_tag'] = df['restaurant_tag'].fillna('Unknown').astype(str)

    df = df.dropna(subset=['price', 'rating'])
    return df


def calculate_correlation(df, col1, col2):
    """
    Calculate correlation between two columns.
    Parameters: datasets, two columns to calculate
    Returns: the correlation coefficient
    """
    correlation = df[col1].corr(df[col2])
    print(f"Correlation between {col1} and {col2}: {correlation:.2f}")
    return correlation


def average_metrics_by_group(df, group_col, target_col):
    """
    Calculate the average of a target column  by another column.
    Parameters: dataset, grouping column, target column
    Returns: dataframe with grouped and averaged data
    """
    grouped = df.groupby(group_col)[target_col].mean().reset_index()
    print(f"Average {target_col} by {group_col}:")
    print(grouped)
    return grouped


def plot_rating_vs_price(df):
    """
    Create a scatterplot of price vs. ratings.
    Parameters: cleaned/processed dataset
    """
    plt.figure(figsize=(8, 6))
    
    sns.scatterplot(x=df['price'], y=df['rating'], alpha=0.5)
    
    sns.regplot(x=df['price'], y=df['rating'], scatter=False, color='blue')
    
    plt.title('Price vs Rating', fontsize=14)
    plt.xlabel('Price Level ($)', fontsize=12)
    plt.ylabel('Rating', fontsize=12)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()


def cuisine_popularity(df):
    """
    Plot a bar chart of the top 20 most popular cuisines by count.
    Parameters: cleaned/processed dataset
    returns: barchart of cuisine popularity
    """
    cuisine_counts = df['restaurant_tag'].value_counts().head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=cuisine_counts.values, y=cuisine_counts.index, \
                palette='viridis')
        
    plt.title('Top 20 Most Popular Cuisines by Restaurant Count')
    plt.xlabel('Number of Restaurants')
    plt.ylabel('Cuisine')
    
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.show()


def neighborhood_analysis(df):
    """
    Barplot for restaurants by neighborhood.
    Parameters: cleaned dataset
    """
    neighborhood_counts = df['restaurant_neighborhood'].value_counts().head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=neighborhood_counts.values, y=neighborhood_counts.index, \
                palette='plasma')
        
    plt.title('Top 15 Neighborhoods by Restaurant Count')
    
    plt.xlabel('Number of Restaurants')
    plt.ylabel('Neighborhood')
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()



def review_distribution(df):
    """
    Plot the distribution of restaurant ratings.
    Parameters: cleaned dataset
    Source: 
    https://www.geeksforgeeks.org/how-to-make-histograms-with-density-plots-\
        with-seaborn-histplot/
    """
    plt.figure(figsize=(8, 6))
    
    sns.histplot(df['rating'], bins=10, kde=True, color='green', alpha=0.6)
    mean_rating = df['rating'].mean()
    
    plt.axvline(mean_rating, color='red', linestyle='--', linewidth=1, \
                label=f'Mean Rating: {mean_rating:.2f}')
        
    plt.title('Distribution of Restaurant Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

def cuisine_popularity_by_neighborhood(df, top_n_neighborhoods=10, \
                                       top_n_cuisines=10):
    """
    Stacked bar chart showing popularity of cuisines in specific neighborhoods.
    Parameters: dataset, number to include of neighborhoods and cuisines
    Source: 
    pandas.crosstab() function in Python
    GeeksforGeeks
    https://www.geeksforgeeks.org › pandas-crosstab-functi...
    """
    top_neighborhoods = df['restaurant_neighborhood'].value_counts().\
        head(top_n_neighborhoods).index
    top_cuisines = df['restaurant_tag'].value_counts().\
        head(top_n_cuisines).index

    filtered_df = df[df['restaurant_neighborhood'].isin(top_neighborhoods) &
                     df['restaurant_tag'].isin(top_cuisines)]

    crosstab = pd.crosstab(filtered_df['restaurant_neighborhood'], \
                           filtered_df['restaurant_tag'])

    crosstab_percent = crosstab.div(crosstab.sum(axis=1), axis=0)

    # Plot the stacked bar chart
    plt.figure(figsize=(12, 8))
    crosstab_percent.plot(kind='bar', stacked=True, colormap='viridis', \
                          figsize=(14, 8), alpha=0.8)

    plt.title('Cuisine Popularity by Neighborhood')
    plt.xlabel('Neighborhood')
    plt.ylabel('Proportion of Restaurants (%)')
    
    plt.xticks(rotation=45, fontsize=12, ha='right')
    plt.yticks(fontsize=12)
    plt.legend(title='Cuisine', bbox_to_anchor=(1.05, 1), loc='upper left', \
               fontsize=10, title_fontsize=12)
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.show()

def average_rating_by_price(df):
    """
    Calculate average rating for each price level and visualize the relationship.
    Parameters: cleaned dataset
    """
    avg_ratings = df.groupby('price')['rating'].mean().reset_index()
    
    print(f"Average Ratings by Price Level:{avg_ratings}")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.barplot(x='price', y='rating', data=avg_ratings, palette='coolwarm')
    
    plt.title('Average Rating by Price Level')
    plt.xlabel('Price Level ($)')
    plt.ylabel('Average Rating')
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()

def growth_opportunity_analysis(df):
    """
    Analyze neighborhoods for growth opportunities based on restaurant 
    density and popularity.
    Parameters: cleaned dataset
    """
    neighborhood_stats = df.groupby('restaurant_neighborhood').\
    agg(total_restaurants=('restaurant_name', 'count'),
    avg_reviews=('review_number', 'mean')
    ).reset_index()
    
    
    neighborhood_stats = neighborhood_stats.sort_values(by='avg_reviews', \
                                                        ascending=False)
    print(f"Neighborhood Growth \
          Opportunity Analysis:{neighborhood_stats}")
    
    plt.figure(figsize=(12, 8))
    
    scatter = sns.scatterplot(
        x='total_restaurants', 
        y='avg_reviews',
        size='avg_reviews', 
        sizes=(20, 300), 
        alpha=0.8, 
        hue='avg_reviews', 
        palette='viridis', 
        data=neighborhood_stats
    )
    
    plt.title('Growth Opportunity Analysis: Neighborhoods vs. Popularity')
    
    plt.xlabel('Number of Restaurants')
    plt.ylabel('Average Reviews per Restaurant')
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    top_opportunities = neighborhood_stats.nlargest(3, 'avg_reviews')
    
    
    for _, row in top_opportunities.iterrows():
        
        plt.text(
            row['total_restaurants'], row['avg_reviews'], 
            row['restaurant_neighborhood'], 
            fontsize=10, weight='bold', color='red'
        )
    
    plt.legend(title='Avg Reviews', fontsize=10, \
               loc='upper right')
        
    plt.tight_layout()
    plt.show()

def perform_pearson_analysis(df, col1, col2):
    """
    Perform Pearson correlation analysis between two columns.
    Parameters:cleaned dataset, two columns
    Returns: pearson correlation and p value
    """
    filtered_df = df.dropna(subset=[col1, col2])

    correlation, p_value = pearsonr(filtered_df[col1], filtered_df[col2])
    print(f"Pearson Correlation between '{col1}' and '{col2}': \
          {correlation:.2f}")
          
    print(f"P-value: {p_value:.5f}")
    
def graph_pearson_analysis(df, col1, col2):
    """
    Graph the Pearson correlation with a regression line.
    Parameters: dataset, name of two columns to analyze
    Source: 
        https://www.geeksforgeeks.org/python-seaborn-regplot-method/
    """

    filtered_df = df.dropna(subset=[col1, col2])
    
    correlation, p_value = pearsonr(filtered_df[col1], filtered_df[col2])
    
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=col1, y=col2, data=filtered_df, alpha=0.6, s=50, \
                    color='teal')
    sns.regplot(x=col1, y=col2, data=filtered_df, line_kws={'color': 'red'}, \
                scatter=False)
    plt.title(f'{col1.capitalize()} vs {col2.capitalize()}\
              Pearson Correlation: {correlation:.2f}, P-value: {p_value:.5f}')
              
    plt.xlabel(col1.capitalize())
    plt.ylabel(col2.capitalize())
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def main():
    #Load and process data
    print("--- Loading Datasets ---")
    reviews, restaurants, neighborhoods = load_all_data()

    print("--- Initial Data Overview ---")
    print("Unique values in 'price':", restaurants['price'].unique())
    print("Unique values in 'rating':", restaurants['rating'].unique())
    print("Unique values in 'restaurant_tag':", \
          restaurants['restaurant_tag'].unique())

    restaurants = preprocess_data(restaurants)

    # --- Dataset Summaries ---
    print("--- Dataset Summaries ---")
    print(f"Reviews Dataset: {len(reviews)} rows, Columns: \
          {list(reviews.columns)}")
    print(f"Restaurants Dataset: {len(restaurants)} rows, \
          Columns: {list(restaurants.columns)}")
    print(f"Neighborhoods Dataset: {len(neighborhoods)} rows, \
          Columns: {list(neighborhoods.columns)}")

    print ( )
    
    # --- Correlation Analysis ---
    print("--- Correlation Analysis ---")
    calculate_correlation(restaurants, 'price', 'rating')
    calculate_correlation(restaurants, 'price', 'review_number')
    calculate_correlation(restaurants, 'rating', 'review_number')

    perform_pearson_analysis(restaurants, 'price', 'rating') 
    perform_pearson_analysis(restaurants, 'price', 'review_number')
    perform_pearson_analysis(restaurants, 'rating', 'review_number')

    graph_pearson_analysis(restaurants, 'price', 'rating')
    graph_pearson_analysis(restaurants, 'price', 'review_number')
    graph_pearson_analysis(restaurants, 'rating', 'review_number')

    print ( )
    
    # --- Visualizations ---
    print("--- Visualizations ---")
    plot_rating_vs_price(restaurants)
    cuisine_popularity(restaurants)
    neighborhood_analysis(restaurants)
    review_distribution(restaurants)
    cuisine_popularity_by_neighborhood(restaurants, top_n_neighborhoods=10, \
                                       top_n_cuisines=10)
        
    print ( )

    # --- Descriptive Statistics ---
    print("--- Descriptive Statistics ---")
    average_metrics_by_group(restaurants, 'restaurant_tag', 'rating')
    average_rating_by_price(restaurants)

    print ( )
    
    # --- Growth Opportunity Analysis ---
    print("--- Growth Opportunity Analysis ---")
    growth_opportunity_analysis(restaurants)
    print ( )
    print("--- End of Analysis ---")

if __name__ == "__main__":
    main()
