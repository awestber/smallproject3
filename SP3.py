import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv('C:\\Users\\alyss\\OneDrive\\Documents\\Python Programs\\Python39\\Annual cause death numbers new.csv')

# Cleaned column names
cleaned_columns = ["Entity", "Code", "Year", "Meningitis_fatalities",
                   "Dementia_fatalities", "Parkinsons_fatalities", "Nutritional_deficiency_fatalities",
                   "Malaria_fatalities", "Drowning_fatalities", "Interpersonal_violence_fatalities",
                   "Maternal_disorder_fatalities", "HIV_AIDS_fatalities", "Drug_disorder_fatalities",
                   "Tuberculosis_fatalities", "Cardiovascular_fatalities", "Lower_respiratory_fatalities",
                   "Neonatal_disorder_fatalities", "Alcohol_disorder_fatalities", "Self_harm_fatalities",
                   "Forces_of_nature_fatalities", "Diarrheal_disease_fatalities", "Environmental_exposure_fatalities",
                   "Neoplasm_fatalities", "Conflict_fatalities", "Diabetes_fatalities", "Chronic_kidney_fatalities",
                   "Poisoning_fatalities", "Protein_energy_malnutrition_fatalities", "Road_injury_fatalities",
                   "Chronic_respiratory_fatalities", "Chronic_liver_fatalities", "Digestive_disease_fatalities",
                   "Fire_fatalities", "Acute_hepatitis_fatalities", "Measles_fatalities"]

# Rename columns
data.columns = cleaned_columns

# List of all fatalities columns (ignoring the target and categorical columns)
fatalities_columns = [
    "Meningitis_fatalities", "Dementia_fatalities", "Parkinsons_fatalities", "Nutritional_deficiency_fatalities",
    "Malaria_fatalities", "Drowning_fatalities", "Interpersonal_violence_fatalities", "Maternal_disorder_fatalities",
    "HIV_AIDS_fatalities", "Drug_disorder_fatalities", "Tuberculosis_fatalities", "Cardiovascular_fatalities",
    "Lower_respiratory_fatalities", "Neonatal_disorder_fatalities", "Alcohol_disorder_fatalities", "Self_harm_fatalities",
    "Forces_of_nature_fatalities", "Diarrheal_disease_fatalities", "Environmental_exposure_fatalities",
    "Neoplasm_fatalities", "Conflict_fatalities", "Diabetes_fatalities", "Chronic_kidney_fatalities", 
    "Poisoning_fatalities", "Protein_energy_malnutrition_fatalities", "Road_injury_fatalities", 
    "Chronic_respiratory_fatalities", "Chronic_liver_fatalities", "Digestive_disease_fatalities", 
    "Fire_fatalities", "Acute_hepatitis_fatalities", "Measles_fatalities"
]

# Drop rows with missing values for the fatalities columns
data_cleaned = data.dropna(subset=fatalities_columns)

# Define a function to categorize the target variable into groups
def group_classes(value):
    """Group fatalities into categories: Low, Medium, and High."""
    if value < 50:
        return "Low"
    elif value < 200:
        return "Medium"
    else:
        return "High"

# Group the target variable (e.g., "Dementia_fatalities")
data_cleaned["Dementia_fatalities_grouped"] = data_cleaned["Dementia_fatalities"].apply(group_classes)

# Set up features and target variable
X = data_cleaned[fatalities_columns]  # All the fatalities columns
y = data_cleaned["Dementia_fatalities_grouped"]  # Grouped target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model
knn.fit(X_train_scaled, y_train)

# Predict on the test data
y_pred_knn = knn.predict(X_test_scaled)

# 1. Accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn:.4f}")

# 2. AUC (For multi-class classification)
y_pred_prob_knn = knn.predict_proba(X_test_scaled)  # All class probabilities
auc_knn = roc_auc_score(y_test, y_pred_prob_knn, multi_class='ovo', average='macro')  # One-vs-One strategy
print(f"KNN AUC: {auc_knn:.4f}")

# 3. Precision, Recall, F1-Score, and Support (from classification report)
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

# Group the data into cause of death categories (based on predefined logic)
def group_by_cause(column_name):
    """Classify each column into a group of diseases or causes."""
    if column_name in ["Meningitis_fatalities", "Malaria_fatalities", "HIV_AIDS_fatalities", 
                       "Tuberculosis_fatalities", "Measles_fatalities", "Diarrheal_disease_fatalities"]:
        return "Infectious Diseases"
    elif column_name in ["Dementia_fatalities", "Parkinsons_fatalities", "Nutritional_deficiency_fatalities",
                          "Cardiovascular_fatalities", "Chronic_respiratory_fatalities", "Chronic_kidney_fatalities",
                          "Diabetes_fatalities", "Neoplasm_fatalities", "Chronic_liver_fatalities", "Digestive_disease_fatalities"]:
        return "Non-Communicable Diseases"
    elif column_name in ["Drowning_fatalities", "Road_injury_fatalities", "Fire_fatalities", 
                         "Self_harm_fatalities", "Poisoning_fatalities"]:
        return "Injuries/Accidents"
    elif column_name in ["Interpersonal_violence_fatalities", "Conflict_fatalities"]:
        return "Violence and Conflict"
    elif column_name in ["Maternal_disorder_fatalities", "Neonatal_disorder_fatalities"]:
        return "Maternal and Neonatal"
    elif column_name in ["Environmental_exposure_fatalities", "Forces_of_nature_fatalities"]:
        return "Environmental"
    return "Other"

# Group fatalities by the cause of death category
grouped_data = {}
for column in fatalities_columns:
    group = group_by_cause(column)
    if group not in grouped_data:
        grouped_data[group] = 0
    grouped_data[group] += data_cleaned[column].sum()

grouped_df = pd.DataFrame(list(grouped_data.items()), columns=['Cause of Death Group', 'Total Fatalities'])

# Plot the results in a bar graph with different colors for each group
palette = {
    "Infectious Diseases": "lightcoral",
    "Non-Communicable Diseases": "lightblue",
    "Injuries/Accidents": "lightgreen",
    "Violence and Conflict": "gold",
    "Maternal and Neonatal": "orchid",
    "Environmental": "lightseagreen",
    "Other": "gray"
}

plt.figure(figsize=(12, 6))
sns.barplot(x='Cause of Death Group', y='Total Fatalities', data=grouped_df, palette=palette)
plt.xlabel("Cause of Death Group")
plt.ylabel("Total Fatalities")
plt.title("Total Fatalities by Cause of Death Group")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()  # Adjust layout for better spacing

# Show the bar graph
plt.show()

# --- Table of Individual NCD Results ---
ncd_columns = [
    "Dementia_fatalities", "Parkinsons_fatalities", "Nutritional_deficiency_fatalities",
    "Cardiovascular_fatalities", "Chronic_respiratory_fatalities", "Chronic_kidney_fatalities",
    "Diabetes_fatalities", "Neoplasm_fatalities", "Chronic_liver_fatalities", "Digestive_disease_fatalities"
]

# Sum the fatalities for each NCD
ncd_fatalities = data_cleaned[ncd_columns].sum()

# Create a DataFrame for the NCD fatalities
ncd_df = pd.DataFrame(ncd_fatalities).reset_index()
ncd_df.columns = ['Disease', 'Total Fatalities']

# Display the table with the exact number of fatalities for NCDs
plt.figure(figsize=(10, 6))
plt.axis('off')  # Hide the axes
plt.table(cellText=ncd_df.values, colLabels=ncd_df.columns, loc='center', cellLoc='center', colLoc='center')
plt.title("Total Fatalities for Non-Communicable Diseases", fontsize=16)

# Show the table
plt.show()
