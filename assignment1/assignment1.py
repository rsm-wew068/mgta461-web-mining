import gzip
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV, cross_validate
from collections import defaultdict
import random
import math

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# Load data
train_interactions = pd.read_csv('assignment1/train_Interactions.csv.gz')
pairs_read = pd.read_csv('assignment1/pairs_Read.csv')
pairs_rating = pd.read_csv('assignment1/pairs_Rating.csv')

# ### Collaborative Filtering with Matrix Factorization (SVD)

# Prepare the data in Surprise's format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_interactions[['userID', 'bookID', 'rating']], reader)

# Perform cross-validation with default SVD model
print("\n--- Baseline Cross-Validation for SVD ---")
baseline_model = SVD()
baseline_cv_results = cross_validate(baseline_model, data, measures=['rmse'], cv=3, verbose=True)
print("Baseline Cross-Validation RMSE:", np.mean(baseline_cv_results['test_rmse']))
# 'n_factors': 1, 'n_epochs': 28, 'lr_all': 0.007, 'reg_all': 0.22
# Grid Search for SVD
param_grid = {
    'n_factors': [0, 1, 2],          # Number of latent factors
    'n_epochs': [28, 29, 30],            # Number of training epochs
    'lr_all': [0.006, 0.007],         # Learning rate for all parameters
    'reg_all': [0.2, 0.21, 0.22]           # Regularization term for all parameters
}
print("\n--- Performing Grid Search for SVD ---")
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(data)

# Print best parameters
print("Best SVD Parameters:", gs.best_params['rmse'])

# Train the tuned SVD model
svd_tuned = SVD(
    n_factors=gs.best_params['rmse']['n_factors'],
    n_epochs=gs.best_params['rmse']['n_epochs'],
    lr_all=gs.best_params['rmse']['lr_all'],
    reg_all=gs.best_params['rmse']['reg_all']
)

trainset = data.build_full_trainset()
svd_tuned.fit(trainset)

# ### Predict Ratings for Rating Task

# Predict ratings for pairs_Rating.csv
pairs_rating['prediction'] = pairs_rating.apply(
    lambda x: svd_tuned.predict(x['userID'], x['bookID']).est, axis=1
)

# Save the rating predictions
pairs_rating[['userID', 'bookID', 'prediction']].to_csv('predictions_Rating.csv', index=False)
print("\nSaved predictions for Rating Task to 'predictions_Rating.csv'.")

# ### Enhanced Read Prediction Task

# Feature Engineering
book_popularity = train_interactions['bookID'].value_counts()
user_activity = train_interactions['userID'].value_counts()

pairs_read['book_popularity'] = pairs_read['bookID'].map(book_popularity)
pairs_read['user_activity'] = pairs_read['userID'].map(user_activity)

# Add SVD-predicted ratings as a feature
pairs_read['predicted_rating'] = pairs_read.apply(
    lambda x: svd_tuned.predict(x['userID'], x['bookID']).est, axis=1
)

# Create a validation set with negative samples
validation_with_negatives = []
for u, b, r in trainset.build_testset():
    # Positive sample
    validation_with_negatives.append((u, b, 1))

    # Negative sample
    while True:
        negative_book = random.choice(list(book_popularity.keys()))
        if negative_book not in trainset.ur[trainset.to_inner_uid(u)]:
            validation_with_negatives.append((u, negative_book, 0))
            break

# Define Prediction Logic for Read Task
def baseline_predict(user, book, popular_books):
    return 1 if book in popular_books else 0

# Optimize Popularity Threshold
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_threshold = 0.5
best_accuracy = 0

for threshold in thresholds:
    # Define popular books for the threshold
    count_threshold = book_popularity.sum() * threshold
    popular_books = book_popularity[book_popularity.cumsum() <= count_threshold].index

    correct_predictions = 0
    total_predictions = len(validation_with_negatives)

    for u, b, actual in validation_with_negatives:
        prediction = baseline_predict(u, b, popular_books)
        if prediction == actual:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Best Popularity Threshold: {best_threshold}, Best Accuracy: {best_accuracy}")

# Final Predictions for Read Task
popular_books = book_popularity[book_popularity.cumsum() <= best_threshold * book_popularity.sum()].index
pairs_read['prediction'] = pairs_read.apply(
    lambda x: baseline_predict(x['userID'], x['bookID'], popular_books), axis=1
)

# Save the read predictions
pairs_read[['userID', 'bookID', 'prediction']].to_csv('predictions_Read.csv', index=False)
print("\nSaved predictions for Read Task to 'predictions_Read.csv'.")