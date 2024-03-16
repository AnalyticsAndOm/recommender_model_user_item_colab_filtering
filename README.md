# Restaurant Recommender System : User-Item Collaborative Filtering

- **Github repository at**: https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering

- **File name**: Restaurant Recommender System Using User-Item Collaborative Filtering.ipynb

- **Dataset Name**: new_ratings, Restaurants

- **Dataset Link**: <a name="dataset_link"></a> https://github.com/AnalyticsAndOm/digital-music-distributor/tree/main/Data%20Preparation/CSV%20files

## Project Description
**Project Title**: Restaurant Recommender System : User-Item Collaborative Filtering.

**Description**: This project aims to develop a restaurant recommender system based on collaborative filtering techniques. Collaborative filtering is a widely used approach for recommendation systems, leveraging user-item interaction data (in this case restaurant ratings) to generate personalized recommendations. In this project, we will utilize a dataset containing user ratings for various restaurants to build a recommendation model

*Short brief on user-item collaborative filtering*: 
Imagine you went to an ice cream parlor with a group of few friends. Your friends have been to this ice cream parlor except for you and you don’t know which flavor to pick. You now rely on the suggestions of your friends who you think have similar taste preference and finally go with the suggestion from the friend who has the highest similarity in taste preference.

Collaborative filtering works in a similar way. It finds users from a dataset who have similar tastes to you and recommends items they liked but you haven't experienced yet. Let’s understand with the help of an illustration. 

<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/The-collaborative-filtering-algorithms-a-user-based-b-item-based.png">
</p>

In the above illustration, both user one and user two liked T-shirt, trouser and cap. This signifies that the two users have high similarity in preferences. However, user 2 liked shoe in addition to the above items. Since user 1 and user 2 have high similarity, and user 1 have not yet encountered the shoe on the shopping website, therefore the website will recommend user 1 the shoe purchased by user 2. 

## Table of Contents
1. [Introduction](#introduction)
    1. [Problem Statement](#sec1p1)
    2. [Objective](#sec1p2)
    3. [Scope](#sec1p3)
    4. [Methodology](#sec1p4)
2. [Description of the dataset](#section2)
    1. [Data Dictionary](#sec2p1)
3. [Procedure and Code Snippets](#section3)
4. [Expected Outcome](#expectedoutcome)
5. [Significance](#significance)
6. [References](#references)

## 1. Introduction <a name="introduction"></a>
- The model was created in Jupyter Notebook under the filename 'Restaurant Recommender System Using User-Item Collaborative Filtering.' To access the file, please download it from the [repository](https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering) and open the 'Restaurant Recommender System Using User-Item Collaborative Filtering.ipynb' file in your Python environment. 

- Alternatively, you can view a static version of the notebook by accessing the [Jupyter Nbviewer](https://nbviewer.org/) by providing the link for the Restaurant Recommender System Using User-Item Collaborative Filtering.ipynb file.

### 1.1. Problem Statement <a name="sec1p1"></a>
In a vast list of restaurants on food aggregator platforms, selecting the ideal restaurant can be challenging due to the absence of personalized recommendations. To address this, develop a restaurant recommender system using user-item collaborative filtering technique which shall analyse user interactions, and offer personalised suggestions based on past online orders, simplifying decision-making for users and enhancing their overall experience.

### 1.2. Objective <a name="sec1p2"></a>

 1. Implement user-item collaborative filtering to generate personalized restaurant recommendations.

 2. Analyze user interactions and preferences to identify similarities and recommend relevant restaurants.

 3. Enhance user satisfaction by providing targeted suggestions aligned with their past ratings.

 4. Improve restaurant visibility and attract new customers by recommending establishments to users likely to enjoy their offerings.

### 1.3. Scope of project <a name="sec1p3"></a>

- Develop a restaurant recommender system using user-item collaborative filtering techniques.

- Utilize user interaction data to train the recommendation model and generate personalized suggestions.

- Model deployment and model evaluation is out of project scope as model deployment would need an existing environment for integration.

### 1.4. Methodology <a name="sec1p4"></a>

**Data Collection**:
- Restaurant related data such as restaurant name, restaurant ID, cuisine types, location, and other relevant attributes, have been obtained from the [Restaurants Dataset | Swiggy](https://www.kaggle.com/datasets/ashishjangra27/swiggy-restaurants-dataset) from Kaggle. The data has been curated by Mr. Ashosh Jangra.

- Restaurant ratings data was not readily available, therefore the new_ratings dataset which is a dummy data has been created for the purpose of this project. This dataset consists of features such as restaurant ID, user ID and the ratings given by each user for restaurants. Rows with blanks or NAN in ratings column is considered as restaurants not previously ordered from.

- The restaurant dataset had no missing values or duplicates. Also, the blanks or NaN in new_ratings dataset need not be imputed or dropped.

**Feature Engineering**
- Only relevant features from the restaurants dataset were selected which include restaurant name, restaurant id.
- The two tables named new_ratings and restaurants were merged into a single table to carry further analysis.
## 2. Description of the dataset <a name="section2"></a>
The dataset consists of total 2 .csv files that provide details regarding restaurants and user ratings. Two tables provide details such as restaurant id, restaurant name, cuisines, veg/non veg, location, user ID, ratings, etc.  

### 2.1. Data Dictionary <a name="sec2p1"></a>
- Below table consists of details of table name and its features. For better understanding of features of each table please refer the data dictionary.


| Table Name | Features | Description |
|------------|----------|-------------|
| Restaurant Name     | rname | Name of the restaurant |
| Restaurant Name     | RID | Unique identifier for each restaurant |
| Restaurant Name     | location | City in which the restaurant is located |
| Restaurant Name     | cuisine | Types of cuisines offered by the restaurant |
| Restaurant Name     | cum_ratings| Cumulative ratings for the retaurant|
| Restaurant Name     | cost for two | Estimated price of ordering food for two people |
| Restaurant Name     | lic_no | License number of each restaurant |
| Restaurant Name     | link | URL that takes user to the landing page of the restaurant |
| Restaurant Name     | menu | Contains a JSON file for the menu |
| Restaurant Name     | address |Address of the restaurant |
| new_ratings      | userID | Unique Identifier for each user |
| new_ratings      | RID | Unique Identifier for each restaurant |
| new_ratings      | value|Ratings given to the restaurant by the user |


## 3. Procedure and Code Snippets <a name="section3"></a>
1. *Importing relevant libraries*: 
The very first step in model development involves importing necessary libraries for data processing, visualization and evaluation purpose. We therefore import the numpy, pandas and scipy libraries for data processing and seaborn for data visualization purpose.

```Python
# Importing Data processing libraries
import pandas as pd
import numpy as np
import scipy.stats

# Importing Visualization Library
import seaborn as sns
```
2. *Reading Data*:
Let’s read in the new_ratings dataset using the .read_csv() method from pandas library.

```Python
# Read in data for restaurant ratings and review the data
ratings=pd.read_csv("E:\\Data Analytics stuff\\Project files\\DS - Recommender Model\\Restaurants\\new_ratings.csv")
ratings.head()

```

<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/1.PNG">
</p>

  There are three columns in the ratings dataset – userID, RID, and value(the ratings for the restaurant). Lets see the total no of unique users, restaurants and rating values.

```Python
# Number of users
print('The ratings dataset has', ratings['userID'].nunique(), 'unique users')

# Number of Restaurants
print('The ratings dataset has', ratings['RID'].nunique(), 'unique restaurants')

# Number of ratings
print('The ratings dataset has', ratings['value'].nunique(), 'unique ratings')

# List of unique ratings
print('The unique ratings are', sorted(ratings['value'].unique()))
```
The dataset has 791013records with 602 unique users, 1314 unique restaurants, and 5 unique ratings with values [1,2,3,4,5].

<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/3.PNG">
</p>

Now that we have analaysed the new_ratings dataset, lets analyse the restaurant names dataset.

```Python
# Read in the restaurants data
restaurants = pd.read_csv("E:\Data Analytics stuff\Project files\DS - Recommender Model\Restaurants\Restaurant Names.xlsx.csv")
restaurants.head()
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/4.PNG">
</p>

Now lets merge the two tables with the help of pd.merge() using RID as the common key so that we have the restaurant names and the userID in the same dataframe. 

```Python
# Merge ratings and restaurants datasets
df = pd.merge(ratings, restaurants, on='RID', how='inner')
df.head()
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/5.PNG">
</p>

3. *Exploratory Data Analysis*:
Now let’s aggregate the dataset with the help of .groupby() method of pandas to check the average ratings and count of ratings against each restaurant name field.

```Python
# Aggregate by restaurants
agg_ratings = df.groupby('rname').agg(mean_rating = ('value', 'mean'),
                                                number_of_ratings = ('value', 'count')).reset_index()
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/6.PNG">
</p>

 Now, with the help of .describe() method let’s check the mean ratings count for the aggregated dataframe.

```Python
#Check the mean rating count for the restaurants for statistical significance
agg_ratings.describe()
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/7.PNG">
</p>

 The mean rating count for the entire dataset is approx 180 ratings.

```Python
# Keep only restaurants with over 180 ratings(mean rating count)
agg_ratings_GT180 = agg_ratings[agg_ratings['number_of_ratings']>180]
agg_ratings_GT180.info()
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/8.PNG">
</p>

 Now lets check what are the popular restaurants.
```Python
# Check popular restaurants
agg_ratings_GT180.sort_values(by='number_of_ratings', ascending=False).head()
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/9.PNG">
</p>

 Let’s merge the aggregate ratings table with the ratings table to keep only restaurants that have more than 180 ratings and drop all other ratings from the dataset.
```Python
# Merge data
df_GT180 = pd.merge(df, agg_ratings_GT180[['rname']], on='rname', how='inner')
df_GT180.info()
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/11.PNG">
</p>

 Now let’s check how many unique restaurants and users are there after we have removed restaurants where ratings are less than 180.
```Python
# Number of users
print('The ratings dataset has', df_GT180['userID'].nunique(), 'unique users')

# Number of restaurants
print('The ratings dataset has', df_GT180['RID'].nunique(), 'unique restaurants')

# Number of ratings
print('The ratings dataset has', df_GT180['value'].nunique(), 'unique ratings')

# List of unique ratings
print('The unique ratings are', sorted(df_GT180['value'].unique()))
```
 After filtering the dataset where restaurants have more than 180 ratings we have 602 unique users and 409 unique restaurants.

<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/12.PNG">
</p>

4. *Create User-Restaurant Matrix*:
 In this step we will create a user- restaurant matrix where the rows represent ratings given by each user for every restaurant, while columns represent the ratings obtained by a particular restaurant from various users.

 This is a very crucial step as this matrix is used to develop the similarity matrix that represents similarity between various users. We will employ the .corrwith() method while there are other methods such as cosine similarity to develop a similarity matrix.

```Python
# Create user-item matrix
matrix = df_GT180.pivot_table(index='userID', columns='rname', values='value')
matrix.head()
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/13.PNG">
</p>

5. *Data Normalization*:
 Although the rating scale is same for all the users, different users use the ratings differently. For example user 1 may rate all the best restaurants with 5 star ratings and all other restaurants with a rating of 2. While user2 may rate all good restaurants with a 4-star rating and all other restaurants as 2-star rating. 

 Therefore, the absolute rating is not important but the relative rating is important to us. To achieve this, we need to normalize the ratings. This can be done by subtracting the average ratings for each user from the ratings awarded by the user for each restaurant.

```Python
# Normalize user-item matrix
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 'rows')
matrix_norm.head()
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/14.PNG">
</p>

6. *Identifying Similar Users*:
 Two of the most widely used methods to calculate the similarity between items are cosine similarity and Pearson correlation. Cosine similarity method represents items of a user item matrix as a vector in an n-dimensional space and then calculates the angle between the vectors. 

 The idea behind it is that similar items (vectors) may have varying magnitude but they have the same direction, which means the angle between the two vectors shall be zero. The values of a similarity matrix obtained through cosine similarity varies from -1 to 1 where a value of -1 represent completely opposite vectors or entirely unrelated items and a value of 1 indicates perfectly related items.

 We shall be using the Pearson correlation method as it can efficiently handle null values and does not need the nulls to be imputed. The Pearson correlation generates a correlation coefficient for items. The value for Pearson’s correlation coefficient varies from -1 to 1 where  value of 0 indicates completely opposite vectors or entirely unrelated items and a value of 1 indicates perfectly related items.

```Python
# User similarity matrix using Pearson correlation
user_similarity = matrix_norm.T.corr()
user_similarity.head()
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/15.PNG">
</p>

 Now lets pick a user in random (say user with userID = 3) and find users similar to this user from the similarity matrix generated in above code. We will first remove userID 2 from the similarity matrix and decide on the number of similar users (say 10) from the similarity matrix.

```Python
# Pick a user ID
picked_userid = 3.0

# Remove picked user ID from the candidate list
user_similarity_wo_picked=user_similarity.drop(index=picked_userid)

# Take a look at the data
user_similarity.head()
```

 The user-based collaborative filtering makes recommendations based on users with similar taste, so we will set a positive threshold of 0.3, which means a user must have a Pearson correlation coefficient of at least 0.3 to be considered as a similar user.
 After setting the number of similar users and similarity threshold, we sort the user similarity value from the highest and lowest, then printed out the most similar users' ID and the Pearson correlation value.
```Python
# Number of similar users
n = 10

# User similarity threashold
user_similarity_threshold = 0.3

# Get top n similar users
similar_users = user_similarity[user_similarity[picked_userid]>user_similarity_threshold][picked_userid].sort_values(ascending=False).head(n)

# Print out top n similar users
print(f'The similar users for user {picked_userid} are', similar_users)
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/users%20similar%20to%203.PNG">
</p>

7. *Removing the unrelated items*:
In order to remove the unrelated items, we will take following actions:
 i. Remove the restaurants that have been ordered from by the target user (user 3)
 ii. Keep only the Restaurants that similar users have ordered from, or remove restaurants that similar users have never ordered from.

```Python
# Restaurants that the target user has ordered from
picked_userid_ordered = matrix_norm[matrix_norm.index == picked_userid].dropna(axis=1, how='all')
picked_userid_ordered
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/16.PNG">
</p>

```Python
# Restaurants that similar users ordered from.
similar_user_restaurants = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1, how='all')
similar_user_restaurants
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/17.PNG">
</p>

 Now, we will drop the restaurants that the target user has ordered from, from the similar_user_restaurants. We can overcome the problem of errors while executing the .drop() method by using the parameter errors=ignore.
```Python
# Remove the ordered from restaurants form the list 
similar_user_restaurants.drop(picked_userid_ordered.columns,axis=1, inplace=True, errors='ignore')

# Take a look at the data
similar_user_restaurants
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/18.PNG">
</p>


8. *Recommend Items*:
 Now we can recommend items to the user. The recommended items are determined by the weighted average of user similarity score and restaurant ratings. The restaurant ratings are weighted by the similarity scores, so the users with higher similarity get higher weights.
 This code loops through items and users to get the item score, rank the score from high to low and pick the top 10 restaurants to recommend to user ID 3.
```Python
# A dictionary to store item scores
item_score = {}

# Loop through items
for i in similar_user_restaurants.columns:
  # Get the ratings for restaurant i
  restaurant_rating = similar_user_restaurants[i]
  # Create a variable to store the score
  total = 0
  # Create a variable to store the number of scores
  count = 0
  # Loop through similar users
  for u in similar_users.index:
    # If the restaurant has rating
    if pd.isna(restaurant_rating[u]) == False:
      # Score is the sum of user similarity score multiply by the restaurant rating
      score = similar_users[u] * restaurant_rating[u]
      # Add the score to the total score for the restaurants so far
      total += score
      # Add 1 to the count
      count +=1
  # Get the average score for the item
  item_score[i] = total / count

# Convert dictionary to pandas dataframe
item_score = pd.DataFrame(item_score.items(), columns=['restaurant', 'restaurant_score'])

# Sort the restaurants by score
ranked_item_score = item_score.sort_values(by='restaurant_score', ascending=False)

# Select top m Restaurants
m = 10
ranked_item_score.head(m)
```
<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/19.PNG">
</p>


9. *Predict Score*:
 In order to predict the user's rating, we need to add the user's average restaurant rating score back to the restaurant score.

```Python
# Average rating for the picked user
avg_rating = matrix[matrix.index == picked_userid].T.mean()[picked_userid]

# Print the average restaurants rating for user 1
print(f'The average restaurant rating for user {picked_userid} is {avg_rating:.2f}')
```
 The average restaurant rating for user 3 is 3.00 so we add back 3.00 to the restaurant score.
```Python
# Calcuate the predicted rating
ranked_item_score['predicted_rating'] = ranked_item_score['restaurant_score'] + avg_rating

# Take a look at the data
ranked_item_score.head(m)
```

<p align="center">
<img src="https://github.com/AnalyticsAndOm/recommender_model_user_item_colab_filtering/blob/main/images/20.PNG">
</p>

## 4. Expected Outcome <a name="expectedoutcome"></a>
 The model is supposed to recommend n top related restaurants when it is provided a user ID in random which is already a part of the dataset.
Note that memory based model face the probem of cold start which means users or items that are added new to the platform are not recommended as there is not sufficient data to establish relevance between users or items.

## 5. Significance <a name="significance"></a>
 1. Recommender systems provides personalized restaurant suggestions based on individual preferences, past dining experiences, enhancing user satisfaction and simplifying decision-making therefor increases sales and customer retention on the platform.
 2. By offering tailored recommendations which are aligned with users' tastes and preferences, recommender systems improve the overall customer experience, leading to higher customer satisfaction and loyalty.
 3. Recommender systems save users time and effort by eliminating the need to manually search through numerous restaurant options.
 4. Recommender System prevents decision paralysis as it reduces the time a user spends in choosing a restaurant and therefore reduces exit ratio on the platform, in other words it improves the conversion ratio.
 5. Recommender system leads to a dual benefit, it not only help the customer to select from new but similar restaurants but it also helps restaurants acquire new customers with relative ease when compared to traditional platforms.
## 6. References <a name="references"></a>
1. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. IEEE Computer, 42(8), 30-37.
2. Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. Proceedings of the 10th international conference on World Wide Web, 285-295.
3. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. 2008 Eighth IEEE International Conference on Data Mining, 263-272.
4. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian personalized ranking from implicit feedback. Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI-09), 452-461.
5. Deshpande, M., & Karypis, G. (2004). Item-based top-N recommendation algorithms. ACM Transactions on Information Systems (TOIS), 22(1), 143-177.
