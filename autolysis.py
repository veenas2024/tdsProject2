# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.2.3",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "openai",
#   "scikit-learn",
#   "python-dotenv",
# ]
# ///
import os
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

file_path= None
prompt= None
chatGptResponse= None
summary_stats= None
correlation_matrix= None
numeric_df = None
token = None
data = None

# Function to perform analysis and generate outputs
def analyze_dataset(file_path):
    print("autolysis:analyzeDataSet started",file_path)
    # Load dataset
    try:
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']  # Add encodings
        for encoding in encodings:
            try:
                data = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                print(f"Error in autolysis:analyzeDataSet encoding: {encoding}")
    except Exception as e:
        print(f"Error in autolysis:analyzeDataSet reading file: {e}")
        return
    # Generate summary statistics
    summary_stats = data.describe()
    
    print("correlation_started")
    # Generate correlation matrix
    try:
        # Select only numeric columns from data set
        numeric_df = data.select_dtypes(include=["number"])
        correlation_matrix = numeric_df.corr()
        # Save correlation matrix as heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix Heatmap")
        plt.savefig("correlation_matrix.png")
        plt.close()
    except Exception as e:
        print(f"Error in correlation: {e}")
        return
    print("correlation_ended")
    
    # Regression analysis
    print("Regression analysis started")
    try:
       if numeric_df.shape[1] < 2:
        raise ValueError("The dataset must have at least two numeric columns for regression.")
        # Choose the column with the highest variance as the target
       variances = numeric_df.var()
       target_column = variances.idxmax()
       # Use all other numeric columns as features
       X = numeric_df.drop(columns=[target_column])
       y = numeric_df[target_column]
       # Handle missing values
       imputer = SimpleImputer(strategy='mean')
       X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
       y = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).flatten(), name=target_column)

       # Scale features
       scaler = StandardScaler()
       X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        # Split the data into training and test sets
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       # Train a linear regression model
       reg_model = LinearRegression()
       reg_model.fit(X_train, y_train)
       # Make predictions
       y_pred = reg_model.predict(X_test)
       mse = mean_squared_error(y_test, y_pred)
       r2 = r2_score(y_test, y_pred)
       coefficients = reg_model.coef_
       intercept = reg_model.intercept_
    except Exception as e:
         print(f"Error in predictors: {e}")
         return
    print("Regression analysis ended")

    #Save regression residual plot
    print("residuals started")
    try:
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True)
        plt.title("Regression Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.savefig("regression_residuals.png")
        plt.close()
    except Exception as e:
        print(f"Error in residuals: {e}")
        return
    print("residuals ended")
    return summary_stats, correlation_matrix, numeric_df, mse, r2, coefficients, intercept
    
    print("autolysis:analyzeDataSet completed")


def call_chatgpt(summary_stats, correlation_matrix, numeric_df, mse, r2, coefficients, intercept):
    print("autolysis:chatGptApiCall started")
    token= os.environ["AIPROXY_TOKEN"]
    if token == None:
        print("Error: AIPROXY_TOKEN environment variable not set or None")
        sys.exit(1)
    # Set OpenAI API key and base URL
    openai.api_key = token
    openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
    # Prepare GPT prompt
    prompt = f"""
    You are an expert data analyst. Please analyze the following dataset.
    ## Summary Statistics
    {summary_stats.to_string()}

    ## Correlation Matrix
    {correlation_matrix.to_string()}

    Perform regression analysis using numeric_df make appropriate feature selection:
    {numeric_df}.
    Include:
    - Coefficients
    - Intercept
    - Mean Squared Error (MSE)
    - R-squared Value

    Provide insights based on the analysis and narrate a story summarizing key findings.
    """
    # Make a ChatCompletion request
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'] 
    except openai.error.AuthenticationError as e:
        print("Authentication failed:", str(e))
        sys.exit(1)
    except openai.error.OpenAIError as e:
        print("OpenAI API error:", str(e))
        sys.exit(1)
    print("autolysis:chatGptApiCall completed")

def write_readme(chatgpt_response):
    print("autolysis:createReadmeFile started")
     # Save the response to README.md
     
    try:
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']  # Add encodings
        for encoding in encodings:
            try:
                with open("README.md", "w", encoding=encoding) as readme_file:
                    readme_file.write(chatgpt_response)
                    readme_file.write("\n\n## Visualizations\n")
                    readme_file.write("### Correlation Matrix\n")
                    readme_file.write("![Correlation Matrix](./correlation_matrix.png)\n\n")
                    break
            except UnicodeDecodeError:
                print(f"Error in autolysis:analyzeDataSet: writing to readme encoding: {encoding}")
        print("ChatGPT response has been written to 'README.md'.")
    except Exception as e:
        print(f"Error in writing ChatGPT response to file: {e}")
        return
    print("autolysis:createReadmeFile completed")

# Main function to parse arguments and call the analysis function
def main():
    print("autolysis:main started")
    try:
        parser = argparse.ArgumentParser(description="Analyze dataset.")
        parser.add_argument("file", type=str, help="Path to the dataset CSV file.")
        args = parser.parse_args()
        print("fine name",args.file)
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} does not exist.")
            return
        file_path=args.file
        print("autolysis:main:file_path",file_path)

        # Analysis stage with the data set provided
        results = analyze_dataset(file_path)

        # ChatGPT stage
        chatgpt_response = call_chatgpt(*results)

        # README generation stage
        write_readme(chatgpt_response)

    except Exception as e:
        print(f"Error in autolysis:main: {e}")
    print("autolysis:main completed")

if __name__ == "__main__":
    main()
