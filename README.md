# Data-Science-Kaggle

# Data Exploration
Sample train.csv
- The train.csv contains over 1 million rows of data and it make the model running take really a long time. So a random sample of 50 thousands rows of data has been used during the model train.
- After getting a really good model (0.63 accuracy), tried to use the whole train.csv to run through the code and get a better accuracy (0.64), which is the final submission

The main data used:
-  the combine text of Summary and Text, and the Score
-  the feature part is also build on the combine text

For PruductId and UserId (tried but not used in final submission):
- Tried and using the mean score of what a same ProductId/UserId got/given. The accuracy was not improved nor decreased, guessing it may because of the 50 thousands data is too small to get a same Id.
- For the final submission, the part of integration for PruductId and UserId has been deleted.


# Feature Extraction / Engineering

Preprocessing of the data before started feature:
- Dropped rows with missing Score from training set, which is what we are predicting and should be in testing set.
- Ensured all scores are integers within the valid range [1, 5].
- Created a new column combined_text by joining Summary and Text, also used .fillna('') to fix missing Summary and Text.

Handcrafted Feature Engineering:
Try to add multiple custom features to capture writing style and sentiment, which includes:
- Text structure: text_length, summary_length, word_count
  - The length of User command may reflect their satisfying or dissatisfying.

- Stylistic cues: count of capitalized words, uppercase words, punctuation (!, ?, ., ,)
  - Capitalized words and punctation (especially ! and ?) show User's emotion.

- Emotion & tone: emoji_count, sentiment_polarity, sentiment_subjectivity (via TextBlob)
  - The emoji part seems not work, maybe because the combined text has be cleaned already.
  - Sentiment is a really important part to show how Use feel about the Product and influence their given Score.
  - Using TextBlob to import those sentiment words (https://textblob.readthedocs.io/en/dev/).

- Ratios: capital_word_ratio, title_word_ratio
  - The ratio of the capital words would show more directly to how User feel because the total text length may vary.

These features were added to both training and testing data to enhance model performance.


# Model Creation and Assumptions

Vectorization:
- Used TfidfVectorizer with n-grams (1–3), max 8000 features, min df=5, max df=0.75, stopwords removal, and sublinear TF scaling.
  - stop_words="english"
    Removes common English stop words like "the", "is", "and", which helps reduce noise in the data. It prevents uninformative words from affecting TF-IDF scores.
  - ngram_range=(1, 3)
    Extracts unigrams (1 word), bigrams (2 words), and trigrams (3 words), which captures more context and improves performance by identifying key phrases rather than just single words.
  - max_features=8000
    Keeps only the top 8000 features (terms) with the highest TF-IDF scores across the corpus. Reduces dimensionality and training time and controls overfitting and computational cost.
  - min_df=5
    Ignores terms that appear in fewer than 5 documents and removes rare or misspelled words that don’t generalize well. It filters out noise and overfitting to rare tokens.
  - max_df=0.75
    Ignores terms that appear in more than 75% of documents which are are too common to be useful. It prevents overemphasis on frequently repeated but uninformative words.
  - sublinear_tf = True
    Applies logarithmic scaling to term frequency and prevents very frequent words from dominating the TF-IDF score. It helps balance extremely frequent terms with medium-frequency ones.
    
- using https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html for finding parameters.


Model:
- Combined TF-IDF features with handcrafted features using scipy.sparse.hstack.
  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.hstack.html
  
- The first model used is DecisionTree, which got a accuracy of about 0.42 with every parameter normalized. After that Used GridSearchCV for fixing parameters and got a accuracy about 0.53(As submission 1 and 2 does). It doesn't work well so RandomForest is used to improve.
  - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- For RandomForest, it reached accuracy of 0.57 with the same feature. It still not that well so decided to try linear regression.
  - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

- Trained a Logistic Regression model (liblinear solver, max_iter=1000) on the combined features.
  - Assumed logistic regression would perform well on this medium-sized, sparse feature space due to its interpretability and efficiency.
  - Finally reached a accuracy of about 0.63.
  - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


# Model Tuning
Hyperparameter grid search
- Used GridSearchCV for max_features ([2000, 5000, 8000]) and ngram_range ([None, (1, 2), (1, 3)])
  - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- Used 5-fold cross-validation:
  - The data was split into 5 folds: training was done on 4 folds and validated on the remaining fold, rotating through all combinations.
  - This ensures the model is evaluated reliably across different splits of the dataset and helps reduce overfitting to a particular train-test split.
  - https://scikit-learn.org/stable/modules/cross_validation.html

- The final result is max_features = 8000 and ngram_range = (1, 3) does the best.
- After decided the best peremeters, used the best and remove the others to make the code clear.

- Tuning opportunities still remain open for other features but the current model is doing well.


# Model Evaluation / Performance
Train Accuracy: 64.85%
Test Accuracy: 64.36%

The model shows no major overfitting, as training and test accuracies are close.

![image](https://github.com/user-attachments/assets/61876c1e-61e3-4a26-98c6-89099fd90cd3)



# Struggles / Issues / Open Questions
- Several reasons why random forests/decision trees are not as good as linear regression:
  - TF-IDF features are linear-friendly
  - Random forests perform poorly in cases of imbalanced and multi-classification
  - This is an ordered multi-classification problem, and ordinary classifiers do not consider the order between labels
 
- Dealing with feature:
  - Some of the features used may not pay a role or decline the accuracy instead, and it may take more time to test and figure out
  - Feature overlap may exists. Possible redundancy between Text and Summary could be investigated.
  - Most scores are 4 or 5, potentially create an Class imbalance. Tried to use balanced parameters in model but it lowered the accuracy instead. May because the testing set is also imbalanced.
  - There may exist sentiment noise, where some reviews use sarcasm, which basic sentiment analysis may misinterpret.

- SVM may do a better performance, but running it takes such long time and is not easy to go through and adjust while building models.
  - time limited and did not use SVM.



