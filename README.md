# MGTA 461: Web Mining & Recommender System

This course focuses on practical applications of machine learning in web mining contexts, including recommendation systems, text classification, and predictive modeling using real-world datasets.

## Homework Assignments

### Homework 2: Model Pipelines and Diagnostics

**Focus**: Machine learning model evaluation and improvement using the Polish Companies Bankruptcy dataset.

**Key Topics**:
- Logistic Regression with regularization (C parameter)
- Class imbalance handling using `class_weight='balanced'`
- Data shuffling and proper train/validation/test splits (50/25/25%)
- Hyperparameter tuning for regularization strength
- Balanced Error Rate (BER) evaluation metric
- Recommendation systems using Jaccard similarity on Goodreads book review data
- Collaborative filtering with both item-item and user-user similarity approaches

**Main Questions**:
1. Train logistic regression with C=1.0 and report accuracy/BER
2. Compare with balanced class weights
3. Implement proper data splitting and cross-validation
4. Regularization pipeline with C values from 10^-4 to 10^4
5. Find optimal C value for best generalization
6. Find most similar items using Jaccard similarity
7. Implement rating prediction using item-item similarity
8. Implement rating prediction using user-user similarity

**Performance Results**:
- **Q1**: Accuracy: 96.57%, BER: 47.67% (without class balancing)
- **Q2**: Accuracy: 69.48%, BER: 30.46% (with class_weight='balanced')
- **Q3**: Training BER: 29.29%, Validation BER: 31.59%, Test BER: 25.86%
- **Q4**: Best C=100 with validation BER: 29.55%
- **Q5**: Best C=100 achieves test BER: 26.27%
- **Q6**: Top similar items to '2767052' with Jaccard similarities up to 41.25%
- **Q7**: Item-item similarity MSE: 1.245 (using Jaccard similarity)
- **Q8**: User-user similarity MSE: 1.252 (using Jaccard similarity)

### Homework 3: Recommendation Systems

**Focus**: Advanced recommendation system algorithms using book interaction data.

**Key Topics**:
- Read prediction (binary classification: will user read a book?)
- Popularity-based baselines with threshold optimization
- Jaccard similarity for collaborative filtering
- Rating prediction using bias models
- Matrix factorization concepts with user/item biases
- Regularization in recommendation systems

**Main Questions**:
1. Evaluate baseline popularity model on validation set with negative sampling
2. Optimize popularity threshold for better performance
3. Implement Jaccard similarity-based prediction
4. Combine popularity and similarity thresholds
5. Upload predictions to Gradescope
6. Fit bias model: rating ≈ α + β_user + β_item with λ=1 regularization
7. Find users with highest/lowest bias values
8. Tune regularization parameter λ for optimal performance

**Performance Results**:
- **Q1**: Baseline accuracy: 71.29% (popularity-based with 50% threshold)
- **Q2**: Optimized threshold: 70% achieves accuracy: 75.35%
- **Q3**: Jaccard similarity accuracy: 71.31% (with threshold optimization)
- **Q4**: Combined approach accuracy: 75.35% (popularity + similarity)
- **Q6**: Bias model MSE: 1.489 (with λ=1 regularization)
- **Q7**: User with highest bias: 'u79275096' (β=0.693), lowest: 'u88024921' (β=-1.818)
- **Q8**: Optimal λ=0.01 achieves MSE: 1.450 (best performance)

### Homework 4: Text Mining and NLP

**Focus**: Text mining and natural language processing using Steam game review data.

**Key Topics**:
- Text preprocessing (punctuation removal, lowercase conversion)
- Bag-of-words feature extraction
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Multiclass classification for game genres
- Word2Vec embeddings
- Logistic regression for text classification

**Main Questions**:
1. Find 1,000 most common words and report top 10 with frequencies
2. Build bag-of-words features and classify game genres
3. Calculate IDF and TF-IDF scores for specific words
4. Additional text mining and NLP applications

**Performance Results**:
- **Q1**: Top 10 most common words with frequencies (e.g., "the", "and", "to", etc.)
- **Q2**: Bag-of-words classification accuracy (TF-IDF + Logistic Regression)
- **Q3**: TF-IDF scores for words: 'character', 'game', 'length', 'a', 'it'
- **Q4**: Additional NLP applications and text mining techniques

## Assignment 1: Recommendation System Implementation

**Focus**: Real-world recommendation system implementation with baseline and advanced approaches.

### Components:

#### `assignment1/baselines.py` - Baseline Implementation
- **Rating Prediction Baseline**: Uses user-specific average ratings with global average fallback
- **Read Prediction Baseline**: Popularity-based binary classification (top 50% of interactions)
- **Simple Approaches**: Establishes performance benchmarks

#### `assignment1/assignment1.py` - Advanced Implementation
- **SVD (Singular Value Decomposition)**: Matrix factorization for rating prediction
- **Grid Search**: Hyperparameter tuning for optimal performance
- **Cross-validation**: 3-fold CV to evaluate model performance
- **Feature Engineering**: Book popularity, user activity, predicted ratings
- **Threshold Optimization**: Tests multiple popularity thresholds (0.1 to 0.9)

#### `assignment1/writeup.txt` - Analysis and Results
- **Read Prediction**: Random Forest model achieves 50.5% accuracy
- **Rating Prediction**: SVD model achieves RMSE of 1.2099
- **Performance Analysis**: Comparison of baseline vs. advanced approaches

**Performance Results**:
- **Baseline Read Prediction**: Popularity-based approach (50% threshold)
- **Advanced Read Prediction**: Random Forest with feature engineering - 50.5% accuracy
- **Baseline Rating Prediction**: User-specific averages with global fallback
- **Advanced Rating Prediction**: SVD with grid search optimization - RMSE: 1.2099
- **Grid Search Results**: Best SVD parameters achieved through cross-validation
- **Feature Engineering Impact**: Book popularity, user activity, and predicted ratings
- **Threshold Optimization**: Tested 0.1-0.9 popularity thresholds for read prediction

## Final Project: Recipe Cuisine Classification System (`final_project/recipes.ipynb`)

**Focus**: Comprehensive machine learning project for classifying recipes by cuisine type using multiple advanced approaches.

### Dataset Characteristics
- **231,637 recipes** with rich metadata
- **102 manually identified cuisines** (63.8% of recipes have cuisine tags)
- **11,674 unique ingredients** requiring sophisticated preprocessing
- **552 unique tags** covering flavors, difficulty, cook time, etc.
- **Multi-label nature**: 55% of recipes have multiple cuisine tags
- **Class imbalance challenge**: American cuisine (31,179 recipes) vs. Namibian cuisine (6 recipes)

### Team Contributions

#### Darren's Section
- **Data Infrastructure**: Creates efficient JSON data structures for fast access
- **User-Item Bias Model**: Implements `rating ≈ α + β_user + β_recipe` from course material
- **Regularization**: Uses λ parameter to prevent overfitting
- **MSE Optimization**: Iterative training with sum of squared errors

#### Rachel's (My) Section

**Primary Role**: Content-Based Recommendation System and SVD Implementation

**SVD Matrix Factorization**:
- **Cuisine Encoding**: Converts cuisine labels to numerical representations
- **Surprise Library Implementation**: Implements SVD for collaborative filtering
- **Hidden Factors**: Captures latent relationships between users and cuisines
- **Matrix Decomposition**: User matrix × Singular values × Recipe matrix
- **Collaborative Filtering**: Leverages user interaction patterns for cuisine prediction

**Content-Based Recommendation System (`final_project/recipes_rachel.ipynb`)**:
- **TF-IDF Vectorization**: Converts recipe steps to numerical feature vectors
- **Cosine Similarity**: Implements similarity-based recipe recommendations
- **Interactive Interface**: Creates user-friendly query system for recipe preferences
- **Knowledge-Based Filtering**: Multi-criteria filtering system including:
  - Ingredient preferences
  - Time constraints (cooking duration)
  - Calorie limits
  - Nutritional analysis (calories, fat, protein, carbohydrates)
  - Rating-based sorting
- **Data Preprocessing**: Comprehensive cleaning and feature engineering
- **Model Persistence**: Saves processed data and models for efficiency

**Technical Achievements**:
- **Interactive Recommendation Engine**: Users can input recipe queries and preferences
- **Multi-Modal Filtering**: Combines content-based and knowledge-based approaches
- **Nutritional Analysis**: Extracts and utilizes 7 nutritional components per recipe
- **Flexible Constraint System**: Handles various user preference combinations
- **Performance Optimization**: Efficient data structures and model caching

#### Akshay's Section
- **Random Forest Classifier**: For cuisine classification using TF-IDF features
- **Cuisine Labeling**: Identifies 28 different cuisines from recipe tags/descriptions
- **Hyperparameter Tuning**: Tests different numbers of estimators (50-1000)
- **Performance Analysis**: Achieves ~57% accuracy for cuisine classification
- **Interactive Prediction**: `predict_cuisine()` function for new recipes

### Technical Approaches Implemented

1. **Baseline Approach**: Predicts most common cuisine (American) for all recipes
2. **Simple Feature-Based Approach**: CuisineScore formula using tags, ingredients, and descriptions
3. **SVD Matrix Factorization**: Latent factor model for user-cuisine preferences
4. **XGBoost with One-vs-Rest**: Multi-label classification with 67% accuracy
5. **Random Forest with TF-IDF**: Literature-based approach achieving ~57% accuracy

### Performance Results

**Overall System Performance**:
- **Best Overall Accuracy**: ~79% (approaching state-of-the-art 85-90%)
- **Class Imbalance Challenge**: American cuisine (31,179 recipes) vs. Namibian cuisine (6 recipes)

**Individual Model Performance**:

**XGBoost with One-vs-Rest**:
- **Accuracy**: 67% overall correctness
- **Macro F1-Score**: 0.29 (poor performance for smaller classes)
- **Weighted F1-Score**: 0.61 (disproportionate influence of larger classes)
- **Hamming Loss**: 0.07 (relatively few incorrect label predictions)
- **Class-Specific Performance**: 
  - "Other" class: 97% recall (dominant class)
  - "Thai" and "Japanese": <10% recall (minority classes)

**Random Forest with TF-IDF**:
- **Accuracy**: ~57% with TF-IDF features
- **Hyperparameter Tuning**: Tested 50-1000 estimators
- **Best Performance**: 500 estimators achieved optimal results
- **Feature Engineering**: TF-IDF vectorization of ingredient lists

**SVD Matrix Factorization**:
- **Collaborative Filtering**: User-cuisine preference modeling
- **Latent Factors**: Captures hidden user-cuisine relationships
- **Matrix Decomposition**: User × Singular Values × Recipe matrices

**Content-Based Recommendation System (Rachel's Implementation)**:
- **TF-IDF Vectorization**: Recipe steps converted to feature vectors
- **Cosine Similarity**: Similarity-based recipe recommendations
- **Knowledge-Based Filtering**: Multi-criteria filtering system
- **Interactive Interface**: User-friendly query system
- **Nutritional Analysis**: 7 nutritional components per recipe
- **Performance**: Efficient recommendation generation with flexible constraints

**Baseline Approach**:
- **Most Common Cuisine**: Predicts "American" for all recipes
- **Performance Benchmark**: Establishes baseline for improvement

### Key Challenges Addressed
- **Class Imbalance**: Severe skew between dominant and minority cuisines
- **Feature Sparsity**: 11,674 unique ingredients requiring Word2Vec embeddings
- **Multi-Label Classification**: 55% of recipes have multiple cuisine tags
- **Data Quality**: Manual cuisine tagging and noise handling

## Technical Stack

- **Python** with Jupyter notebooks
- **scikit-learn**: Random Forest, TF-IDF, cosine similarity, logistic regression
- **Surprise**: SVD collaborative filtering
- **XGBoost**: Gradient boosting for multi-label classification
- **pandas**: Data manipulation and preprocessing
- **Word2Vec**: Ingredient embedding generation
- **matplotlib**: Performance visualization
- **Real datasets**: Goodreads book interactions, Steam game reviews, Polish bankruptcy data, Recipe cuisine classification

## Course Progression

This course demonstrates a progressive learning approach:
1. **Homework 2**: Basic ML concepts, regularization, simple recommendation algorithms
2. **Homework 3**: Advanced recommendation systems, matrix factorization, bias models
3. **Assignment 1**: Real-world recommendation system implementation
4. **Homework 4**: Text mining and NLP applications
5. **Final Project**: Comprehensive multi-approach machine learning system

The course emphasizes practical application of machine learning concepts to real-world problems in web mining, recommendation systems, and text analysis.