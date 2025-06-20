# ------------------------------------------
# 📦 Step 1: Load & Clean Data for Tableau
# ------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load original CSV
file_path = '/Users/devraj/Desktop/Bank Telemarketing Campaign Analytics Dashboard/bank-full.csv'
df = pd.read_csv(file_path, sep=';')

# Map 'yes'/'no' to binary
df['subscribed'] = df['y'].map({'yes': 1, 'no': 0})
df.drop(columns=['y'], inplace=True)

# Add age groups
bins = [17, 25, 35, 45, 55, 65, 100]
labels = ['18–25', '26–35', '36–45', '46–55', '56–65', '65+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

# Order months
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)

# Export cleaned CSV for Tableau
output_file = '/Users/devraj/Desktop/Bank Telemarketing Campaign Analytics Dashboard/bank_cleaned_for_tableau.csv'
df.to_csv(output_file, index=False)
print("✅ Cleaned dataset ready for Tableau:", output_file)

# ------------------------------------------
# 🤖 Step 2: Machine Learning Pipeline
# ------------------------------------------

# Drop leakage and derived columns
df_ml = df.drop(columns=['duration', 'age_group'])  # 'duration' is data leakage

# Split features and label
X = df_ml.drop(columns=['subscribed'])
y = df_ml['subscribed']

# Identify categorical columns (including pandas 'category' dtype)
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 🛠️ Fix: Convert all categorical columns to string before encoding
for col in categorical_cols:
    X[col] = X[col].astype(str)

# OneHotEncoder for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Leave numeric columns as is
)

# Fit and transform the features
X_encoded = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))


# ------------------------------------------
# 💾 Step 3: Save Model and Encoder for Streamlit
# ------------------------------------------
import joblib

joblib.dump(model, 'rf_model.pkl')
joblib.dump(preprocessor, 'encoder.pkl')
print("✅ Model and encoder saved! Ready for Streamlit app.")


#✅ Quick Definitions Summary:
#Metric	What it tells you
#Precision	How accurate are my positive predictions?
#Recall	How many actual positives did I catch?
#F1-score	Balance between precision and recall
#Accuracy	Overall how many predictions were correct

#🎓 If You're Using This in a Resume or Interview:
#“I built a full-cycle project analyzing over 45,000 customer interactions from a bank marketing campaign. I developed a Tableau dashboard to visualize conversion trends by contact method, job, age group, and month, and built a machine learning model (Random Forest) that identified key factors driving subscription rates. This enabled data-driven recommendations on campaign timing and customer targeting.”

