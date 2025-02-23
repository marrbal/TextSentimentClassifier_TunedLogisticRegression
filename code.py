# Split data into features and labels
X_train = train_data['text']
y_train = train_data['label']
X_test = test_data['text']
y_test = test_data['label']

# Convert text data to TF-IDF features
tfidf = TfidfVectorizer(max_features=60000, ngram_range=(1, 3), stop_words='english', min_df=2, max_df=0.85)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Define Logistic Regression model
logistic = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')

# Perform hyperparameter tuning with finer grid
param_grid = {
    'C': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
}
grid_search = GridSearchCV(logistic, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)

# Best model after hyperparameter tuning
best_logistic = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Cross-validation for validation
cv_scores = cross_val_score(best_logistic, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")

# Train the best model
best_logistic.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = best_logistic.predict(X_test_tfidf)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
