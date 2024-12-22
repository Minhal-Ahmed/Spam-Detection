### Email Spam Detection Using Machine Learning

---

# Spam Detection with Machine Learning

This project demonstrates a machine learning-based approach to classify emails as spam or ham (non-spam). Using text preprocessing, feature extraction, and a Random Forest Classifier, the model achieves efficient spam detection. The project also includes evaluation metrics to measure the model's performance.

---

## Features

- **Text Preprocessing**:
  - Converts text to lowercase.
  - Removes punctuation.
  - Filters common English stopwords.
  - Stems words using the PorterStemmer algorithm.
  
- **Feature Extraction**:
  - Uses `CountVectorizer` to transform text into numerical representations.
  
- **Model Training and Testing**:
  - Splits the dataset into training and testing sets.
  - Utilizes a Random Forest Classifier for spam classification.

- **Evaluation Metrics**:
  - Computes accuracy, precision, recall, F1 score, and ROC-AUC score.
  
- **Email Classification**:
  - Allows classification of new email text as spam or ham.

---

## Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `wordcloud`
  - `scikit-learn`
  - `nltk`

Install the required libraries with:
```bash
pip install numpy pandas matplotlib seaborn wordcloud scikit-learn nltk
```

---

## Dataset

The project uses a CSV file, `spam_ham_dataset.csv`, with the following columns:
- `text`: The content of the email.
- `label_num`: Labels (`0` for ham, `1` for spam).

Ensure the dataset is placed in the project directory.

---

## Usage

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Preprocess the Data**:
   - The script preprocesses email text to remove noise.

3. **Train the Model**:
   - The Random Forest Classifier is trained on the preprocessed data.

4. **Evaluate the Model**:
   - The script computes performance metrics on the test data.

5. **Classify New Emails**:
   - Input an email to classify it as spam or ham.

---

## How to Run

1. Place the `spam_ham_dataset.csv` file in the project directory.
2. Run the script:
   ```bash
   python <script_name>.py
   ```
   Replace `<script_name>` with the name of the script file.

---

## Customization

- **Adjust Train-Test Split**:
  Modify `test_size` in `train_test_split` to change the data split ratio.

- **Change Classifier**:
  Replace `RandomForestClassifier` with another model, such as SVM or Naive Bayes.

- **Enhance Preprocessing**:
  Add more steps like lemmatization or domain-specific stopwords.

---

## Future Work

- Integrate TF-IDF for feature extraction.
- Add deep learning models to enhance classification accuracy.
- Build a user-friendly GUI or API for easier usage.
- Implement hyperparameter tuning for the classifier.

---

## License

This project is licensed under the [MIT License](LICENSE).

---


## Acknowledgments

Special thanks to the authors of the dataset used in this project.

--- 


