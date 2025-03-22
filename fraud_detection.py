# fraud_detection.py

# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to generate synthetic data
def generate_synthetic_data(n_samples=1000, fraud_ratio=0.1):
    n_fraud = int(n_samples * fraud_ratio)
    n_non_fraud = n_samples - n_fraud

    # Non-fraud data
    non_fraud_data = {
        'Total_Debt_to_Total_Assets': np.random.normal(0.4, 0.1, n_non_fraud),
        'Current_Ratio': np.random.normal(1.5, 0.3, n_non_fraud),
        'Net_Profit_Margin': np.random.normal(0.1, 0.05, n_non_fraud),
        'Return_on_Assets': np.random.normal(0.05, 0.02, n_non_fraud),
        'Revenue_Growth_Rate': np.random.normal(0.05, 0.02, n_non_fraud),
        'Expense_Growth_Rate': np.random.normal(0.04, 0.015, n_non_fraud),
        'Asset_Turnover': np.random.normal(0.8, 0.2, n_non_fraud),
        'Accounts_Receivable_Turnover': np.random.normal(6.0, 1.0, n_non_fraud),
        'Fraud': [0] * n_non_fraud
    }

    # Fraud data
    fraud_data = {
        'Total_Debt_to_Total_Assets': np.random.normal(0.6, 0.15, n_fraud),
        'Current_Ratio': np.random.normal(0.8, 0.2, n_fraud),
        'Net_Profit_Margin': np.random.normal(0.15, 0.07, n_fraud),
        'Return_on_Assets': np.random.normal(0.03, 0.01, n_fraud),
        'Revenue_Growth_Rate': np.random.normal(0.15, 0.05, n_fraud),
        'Expense_Growth_Rate': np.random.normal(0.02, 0.01, n_fraud),
        'Asset_Turnover': np.random.normal(0.5, 0.15, n_fraud),
        'Accounts_Receivable_Turnover': np.random.normal(4.0, 1.5, n_fraud),
        'Fraud': [1] * n_fraud
    }

    # Combine into DataFrame
    df_non_fraud = pd.DataFrame(non_fraud_data)
    df_fraud = pd.DataFrame(fraud_data)
    df = pd.concat([df_non_fraud, df_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# Main pipeline function
def run_pipeline(n_samples=1000, fraud_ratio=0.1, n_estimators=100, max_depth=10):
    print(f"### Memulai Pipeline dengan {n_samples} Sampel dan Rasio Penipuan {fraud_ratio}")

    # Step 1: Generate Synthetic Data
    print("#### Langkah 1: Membuat Data Sintetis")
    df = generate_synthetic_data(n_samples, fraud_ratio)
    print(df.head())
    print(f"**Jumlah Sampel:** {n_samples}, **Jumlah Penipuan:** {int(n_samples * fraud_ratio)}")

    # Visualize data distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Revenue_Growth_Rate', hue='Fraud', bins=30, kde=True)
    plt.title('Distribusi Tingkat Pertumbuhan Pendapatan: Penipuan vs Non-Penipuan')
    plt.savefig('revenue_growth_distribution.png')
    plt.close()

    # Step 2: Preprocessing
    print("#### Langkah 2: Pra-pemrosesan Data")
    X = df.drop('Fraud', axis=1)
    y = df['Fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"**Data Pelatihan:** {X_train.shape[0]} sampel, **Data Pengujian:** {X_test.shape[0]} sampel")

    # Step 3: Train Model
    print("#### Langkah 3: Pelatihan Model Random Forest")
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    print(f"Model dilatih dengan {n_estimators} pohon dan kedalaman maksimum {max_depth}")

    # Step 4: Evaluation
    print("#### Langkah 4: Evaluasi Model")
    y_pred = rf_model.predict(X_test_scaled)
    print("Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriks Kebingungan')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Step 5: Feature Importance
    print("#### Langkah 5: Analisis Pentingnya Fitur")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(feature_importance)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Pentingnya Fitur dalam Deteksi Penipuan')
    plt.savefig('feature_importance.png')
    plt.close()

    # Step 6: Save Results
    print("#### Langkah 6: Menyimpan Hasil")
    results_df = X_test.copy()
    results_df['Actual_Fraud'] = y_test
    results_df['Predicted_Fraud'] = y_pred
    results_df['Fraud_Probability'] = rf_model.predict_proba(X_test_scaled)[:, 1]

    # Save to CSV
    results_df.to_csv('fraud_detection_results.csv', index=False)
    print("**File hasil telah disimpan sebagai 'fraud_detection_results.csv'**")

# Main function to run the pipeline
def main():
    # Set default parameters
    n_samples = 1000
    fraud_ratio = 0.1
    n_estimators = 100
    max_depth = 10

    # Run the pipeline
    run_pipeline(n_samples, fraud_ratio, n_estimators, max_depth)

if __name__ == "__main__":
    main()