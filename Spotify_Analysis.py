# Import Libraries
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logger_setup
import logging

# Create Buffer
buffer = io.StringIO()

# Load and Read Data and EDA
df = pd.read_csv("./Raw Dataset/spotify_data.csv")
logging.info("Data loaded Successfully.")
logging.info("Data Analysis Started...")

logging.info(f"Top 5 Data Sample:\n {df.head()}")
logging.info(f"Bottom 5 Data Sample:\n {df.tail()}")
logging.info(f"Data Shape: {df.shape}")

# Store the data information into buffer
df.info(buf=buffer)

# Extract the information from the buffer into String
info_string = buffer.getvalue()

logging.info(f"Dataset Information: \n {info_string}")
logging.info(f"Dataset Summary: \n {df.describe()}")
logging.info(f"Dataset contains Null/NAN values:\n {df.isnull().sum()}")
logging.info(f"Dataset contains Duplicates: {df.duplicated().sum()}")
logging.info(f"Time Duration between {df['year'].min()} to {df['year'].max()}")
logging.info(f"Total Artist: {df['artist_name'].nunique()}")
logging.info(f"Number of songs sang by Artist:\n {df['artist_name'].value_counts()}")
logging.info(f"Total Number of Genre: {df['genre'].nunique()}")
logging.info(f"Total number of songs in a perticular genre:\n {df['genre'].value_counts()}")

logging.info("Data Analysis Completed !")
logging.info("="*100)

# Data Cleaning
logging.info("Data Cleaning Started...")
df = df.drop_duplicates(subset=['artist_name', 'track_name'])
df = df.dropna(subset=['genre','popularity'])
logging.info("Data Cleaning Done !")
logging.info("="*100)

# Spotify Finding some Questions:
"""
Q.1 - Which genre is in top by average popularity?
Q.2 - How has been changing the Audio Feature (danceability, energy, valence) with time?
Q.3 - Correlation between Popularity vs Audio Features?
Q.4 - Which are the top artists as per cumulative Popularity?
Q.5 - Is there change in time duration of songs with time or not?
"""

# Let's Find these Questions Answer:
# Q.1 - Top 10 genres by average popularity:
top_genre = df.groupby('genre')['popularity'].mean().sort_values(ascending=False).head(10)
logging.info(f"Top 10 Genre according to avg Popularity:\n {top_genre}")

# Create Graph
plt.figure(figsize=(8,6))
top_genre.plot(kind='bar', color='mediumvioletred')
plt.title("Top 10 Genre by Average Popularity")
plt.tight_layout()
plt.savefig("./Graphs/genre_popularity.png")
plt.show()

# Q.2 - Audio feature trends over years:
year_on_year_audio_trend = df.groupby('year')[['danceability','energy','valence']].mean()
logging.info(f"Audio Feature trend Year on Year:\n {year_on_year_audio_trend}")

# Create Graph
plt.figure(figsize=(10,6))
year_on_year_audio_trend.plot()
plt.title("Audio Feature Trends Over Year")
plt.ylabel("Average Score (0-1)")
plt.tight_layout()
plt.savefig("./Graphs/audio_trends.png")
plt.show()

# Q.3 - Correlation: Audio features vs Popularity:
corr = df[['popularity','danceability','energy','valence','tempo','duration_ms']].corr()
logging.info(f"Correlation between Audio Features and Popularity: \n {corr}") 

# Create Graph
plt.figure(figsize=(7,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Audio features vs Popularity Correlation")
plt.tight_layout()
plt.savefig("./Graphs/spotify_correlation.png")
plt.show()

# Q.4 - Top 10 Artists by cummulative popularity:
top_artists = df.groupby('artist_name')['popularity'].sum().sort_values(ascending=False).head(10)
logging.info(f"Top 10 Artists by cummulative Popularity:\n {top_artists}")

# Create Graph
plt.figure(figsize=(8,6))
top_artists.plot(kind='barh', color='indigo')
plt.title("Top 10 Artists by Cummulative Popularity")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("./Graphs/top_artists.png")
plt.show()

# Q.5 - Average song duration trend over years:
duration_trend = df.groupby('year')['duration_ms'].mean()/60000 #convert ms to minutes
logging.info(f"Duration Trend Year on Year:\n {duration_trend}")

# Create Graph
plt.figure(figsize=(9,5))
duration_trend.plot(marker="o", color='darkorange')
plt.title("Average Song Duration Trend Over Year in Minutes")
plt.ylabel('Duration (Minutes)')
plt.tight_layout()
plt.savefig("./Graphs/duration_trend.png")
plt.show()