import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Function to load and clean the dataset
def load_and_clean_data(file_path):
    # Load the dataset
    titanic = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    titanic.drop(columns=['PassengerId','Name','Ticket', 'Cabin'], axis=1, inplace=True)
    
    # Drop rows with missing values in 'Embarked' column
    titanic.dropna(subset=['Embarked'], inplace=True)
    
    # Fill missing values in 'Age' column with mean
    titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)
    
    # Encoding categorical variables
    titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
    titanic['Embarked'] = titanic['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    
    return titanic


# Function to remove outliers
def remove_outliers(df):
    # Calculate IQR and bounds for outliers
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter DataFrame to remove outliers
    df = df[(df >= lower_bound) & (df <= upper_bound)]
    
    return df

# Function to visualize data
def visualize_data(df):
    # Plotting Histogram
    df.hist(figsize=(10, 10), color='#E0B0FF')
    plt.show()

    # Plotting countplots for categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        plt.figure(figsize=(5, 5))
        sns.countplot(x=df[col], palette='magma')
        plt.xlabel(col)
        plt.show()

    # Plotting boxplot for dataset
    sns.boxplot(data=df, palette='magma')
    plt.title('Boxplot for dataset')
    plt.show()

    # Plotting correlation heatmap
    sns.heatmap(df.corr(), annot=True, cmap='Pastel1')
    plt.title('Correlation Plot')
    plt.show()

    # Plotting pairplot
    sns.pairplot(df, hue='Survived', palette='bwr')
    plt.show()

# Main function to execute the script
def main():
    file_path = 'Titanic-Dataset.csv'
    titanic = load_and_clean_data(file_path)
    visualize_data(titanic)

if __name__ == "__main__":
    main()
