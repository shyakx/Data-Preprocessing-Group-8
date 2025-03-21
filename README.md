# Data Preprocessing Project

## Overview
This project focuses on preprocessing, augmenting, and merging two real-world datasets to predict customer spending behavior using machine learning. The project is divided into three main parts:
1. **Data Augmentation on CSV Files**
2. **Merging Datasets with Transitive Properties**
3. **Data Consistency and Quality Checks**

Additionally, a **Bonus Challenge** involves training a machine learning model to predict customer spending.

---
GitHub Repository Structure
```
customer-spending-prediction/
│
├── README.md                             
├── Data Preprocessing Project_Report.pdf                          
│
├── data/
│   ├── customer_transactions.csv         
│   ├── customer_social_profiles.csv      
│   ├── id_mapping.csv                    
│   ├── customer_transactions_augmented.csv  
│   ├── final_customer_data_[group8].csv  
│   └── final_dataset_ready_[group8].csv  
│
├── notebooks/
│   └── customer_spending_prediction.ipynb  
│
├── models/                                
│   └── customer_spending_model.pkl       
```
---

## Datasets
1. **`customer_transactions.csv`**
   - Contains customer transaction data, including:
     - `customer_id_legacy`: Unique customer ID.
     - `transaction_id`: Unique transaction ID.
     - `purchase_amount`: Amount spent in the transaction.
     - `purchase_date`: Date of the transaction.
     - `product_category`: Category of the product purchased.
     - `customer_rating`: Customer rating for the transaction.

2. **`customer_social_profiles.csv`**
   - Contains customer social media activity data, including:
     - `customer_id_new`: Unique customer ID (different from `customer_id_legacy`).
     - `social_media_activity`: Social media activity score.
     - `purchase_interest_score`: Customer's interest in purchasing.

3. **`id_mapping.csv`**
   - Maps `customer_id_legacy` (from transactions) to `customer_id_new` (from social profiles).

---

## Project Steps

### Part 1: Data Augmentation on CSV Files
1. **Load the Dataset:**
   - Load `customer_transactions.csv`.
2. **Data Cleaning & Handling Missing Values:**
   - Handle missing values in the `customer_rating` column using mean imputation.
   - used the predictive modelling technique to fill in missing values as well
3. **Data Augmentation Strategies:**
   - Add random noise to numerical columns (`purchase_amount`, `customer_rating`).
   - Apply log transformation to `purchase_amount`.
   - Generated synthetic data using K-means Clustering.
4. **Export the Augmented Data:**
   - Save the augmented dataset as `customer_transactions_augmented.csv`.

---

### Part 2: Merging Datasets with Transitive Properties
1. **Load the Datasets:**
   - Load `customer_transactions_augmented.csv`, `customer_social_profiles.csv`, and `id_mapping.csv`.
2. **Perform a Complex Merge:**
   - Use `id_mapping.csv` to link `customer_id_legacy` and `customer_id_new`.
   - Merge the datasets based on transitive relationships.
3. **Feature Engineering:**
   - Create a **Customer Engagement Score** using `purchase_amount` and `purchase_interest_score`.
   - Calculate **moving averages** of `purchase_amount`.
4. **Export the Final Preprocessed Data:**
   - Save the merged and feature-engineered dataset as `final_customer_data_[group 8].csv`.

---

### Part 3: Data Consistency and Quality Checks
1. **Data Integrity Checks:**
   - Check for duplicate entries and missing values.
   - Validate that all customer transactions match a valid social profile.
2. **Statistical Summarization:**
   - Generate summary statistics for numerical columns.
   - Visualize the distribution of `purchase_amount`.
3. **Feature Selection for Machine Learning:**
   - Identify highly correlated features using a correlation heatmap.
   - Select the top 10 most important features using feature importance.
4. **Export the Final Dataset:**
   - Save the final dataset as `final_dataset_ready_[group 8].csv`.

---

### Bonus Challenge: Predict Customer Spending
1. **Prepare the Data:**
   - Load `final_dataset_ready_[group 8].csv`.
   - Split the data into training and testing sets.
   - Scale numerical features (if necessary).
2. **Train a Machine Learning Model:**
   - Use **Random Forest Regression** to predict `purchase_amount`.
3. **Evaluate the Model:**
   - Calculate **Mean Squared Error (MSE)** and **R² Score**.
   - Visualize predictions vs actual values.
4. **Save the Model:**
   - Save the trained model as `customer_spending_model.pkl`.

---

## How to Run the Code
1. **Open the Google Colab Notebook:**
   - Open the `customer_spending_prediction.ipynb` notebook in Google Colab.
2. **Upload the Datasets:**
   - Upload `customer_transactions.csv`, `customer_social_profiles.csv`, and `id_mapping.csv` to the Colab environment.
3. **Run the Code Cells:**
   - Execute each code cell in the notebook sequentially.
4. **Download the Output Files:**
   - Download the augmented, merged, and final datasets (`customer_transactions_augmented.csv`, `final_customer_data_[groupNumber].csv`, `final_dataset_ready_[groupNumber].csv`).
   - Download the trained model (`customer_spending_model.pkl`).

---

## Dependencies
The project requires the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`

Install the dependencies using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```
Results
Data Augmentation:

Expanded the transaction dataset using synthetic data.

Merged Dataset:

Successfully merged transaction and social profile datasets using ID mapping.

Machine Learning Model:

Trained a Random Forest Regression model to predict customer spending.

Achieved an R² Score of 0.85 and MSE of 0.195.

Demo Presentation: https://youtu.be/me_e_VNfxLo

## Contributors

&#x20;

- **Bernice Awinpang Akudbilla** – PART 1
- **Kevin Kenny Mugisha**  - PART 2
- **Steven SHYAKA** – PART 3 & BONUS CHALLENGE

---
