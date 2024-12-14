# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.2.3",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "openai==0.28.0",
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

#file_path= None
prompt = None
llm_Response = None
summary_stats = None
correlation_matrix = None
numeric_df = None
token = None
data = None

# Function to perform analysis and generate outputs
def analyze_dataset(file_path):
    """
    Analyze the dataset, perform preprocessing, and return results required for analysis.

    Parameters:
        file_path (str): Path to the dataset CSV file.

    Returns:
        tuple: Summary statistics, correlation matrix, numeric DataFrame, regression results
    """
    print("autolysis:analyzeDataSet started",file_path)
    # Load dataset
    try:
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']  # Add encodings
        for encoding in encodings:
            try:
                data = pd.read_csv(file_path, encoding = encoding)
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
        # Choose the columns with the highest variance as the target
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
    
    #Returning data analysis results
    return summary_stats, correlation_matrix, numeric_df, {
        "mse": mse,
        "r2": r2,
        "coefficients": coefficients.tolist(),
        "intercept": intercept
    }
    
    print("autolysis:analyzeDataSet completed")


def call_llm(summary_stats, correlation_matrix, numeric_df, regression_results):
    """
    Calls a language model to analyze dataset statistics and regression results,
    and generates a narrative summarizing key findings and implications.

    Parameters:
        summary_stats (DataFrame): Summary statistics of the dataset.
        correlation_matrix (DataFrame): Correlation matrix of the dataset.
        numeric_df (DataFrame): Numeric data used for regression analysis.
        regression_results (dict): Dictionary containing 'mse', 'r2', 'coefficients', and 'intercept'.

    Returns:
        dict: Dictionary containing narrative and visualization suggestions.
    """
    print("autolysis:call_llm started")

    token = os.environ.get("AIPROXY_TOKEN")
    if not token:
        print("Error: AIPROXY_TOKEN environment variable not set or None")
        sys.exit(1)

    openai.api_key = token
    openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

    # Dynamically adjust the prompt based on dataset characteristics
    dynamic_features = numeric_df.columns.tolist()
    prompt = f"""
    You are an expert data analyst. Analyze the following dataset and provide a comprehensive narrative.

    ## Dataset Overview
    ### Summary Statistics
    {summary_stats.to_string()}

    ### Correlation Matrix
    {correlation_matrix.to_string()}

    ## Regression Analysis
    - Dataset: {numeric_df.shape[0]} rows, {numeric_df.shape[1]} columns
    - Features: {', '.join(dynamic_features)}
    - Coefficients: {regression_results['coefficients']}
    - Intercept: {regression_results['intercept']}
    - Mean Squared Error (MSE): {regression_results['mse']}
    - R-squared Value: {regression_results['r2']}

    ### Tasks:
    1. Summarize the dataset statistics and key relationships from the correlation matrix.
    2. Identify the most significant features based on regression coefficients.
    3. Interpret the regression results:
       - Discuss the significance of the coefficients.
       - Highlight the predictive power of the model using R-squared and MSE.
    4. Provide actionable insights based on the analysis.
    5. Craft a narrative that:
       - Highlights key findings.
       - Discusses implications for decision-making.
       - Suggests potential next steps or areas for further investigation.
    """
    narrative = ""
    visualizations = ""

    try:
        # First LLM call: Dataset analysis and narrative generation
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert data analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        narrative = response['choices'][0]['message']['content']

    except openai.error.OpenAIError as e:
        print("OpenAI API error during narrative generation:", str(e))
        narrative = "Failed to generate narrative. Proceeding with fallback analysis."

    try:
        # Second LLM call: Generate visualization suggestions if required
        vis_prompt = f"""
        Based on the analysis, suggest three key visualizations that can help illustrate the findings effectively.
        - Dataset: {numeric_df.shape[0]} rows, {numeric_df.shape[1]} columns
        - Key features: {', '.join(dynamic_features)}
        """

        vis_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data visualization expert."},
                {"role": "user", "content": vis_prompt}
            ]
        )

        visualizations = vis_response['choices'][0]['message']['content']

    except openai.error.OpenAIError as e:
        print("OpenAI API error during visualization suggestions:", str(e))
        visualizations = "Failed to generate visualization suggestions. Proceeding with fallback visualizations."

    # Fallback visualization generation if LLM fails
    if not visualizations:
        try:
            plt.figure(figsize=(8, 6))
            pd.plotting.scatter_matrix(numeric_df, alpha=0.2, figsize=(10, 10))
            plt.suptitle("Scatter Matrix of Numeric Features")
            plt.savefig("scatter_matrix.png")
            plt.close()

            visualizations = "Scatter matrix saved as 'scatter_matrix.png'."
        except Exception as e:
            print("Error during fallback visualization generation:", str(e))
            visualizations = "Visualization generation failed. Please check the input data."

    write_readme(narrative, visualizations)

    return {
        "narrative": narrative,
        "visualizations": visualizations
    }

def write_readme(narrative, visualizations):
    """
    Write a README file summarizing the narrative and visualizations.

    Parameters:
        narrative (str): Narrative generated by the LLM or fallback analysis.
        visualizations (str): Description of visualizations or fallback visuals.
    """
    print("autolysis:createReadmeFile started")

    if not narrative:
        narrative = "Analysis narrative could not be generated. Please refer to the visualizations for insights."

    if not visualizations:
        visualizations = "Visualizations could not be generated. Please check the input data."

    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']

    for encoding in encodings:
        try:
            with open("README.md", "w", encoding=encoding) as readme:
                readme.write("# Analysis Report\n\n")
                readme.write("## Narrative\n\n")
                readme.write(narrative + "\n\n")
                readme.write("## Visualizations\n\n")
                readme.write(visualizations + "\n")

                # Include a placeholder for the correlation matrix image
                readme.write("### Correlation Matrix\n\n")
                readme.write("![Correlation Matrix](./correlation_matrix.png)\n\n")

            print(f"README.md successfully written using encoding: {encoding}")
            break
        except Exception as e:
            print(f"Error writing README with encoding {encoding}: {e}")

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
        
        file_path = args.file
        print("autolysis:main:file_path",file_path)

        # Analysis stage with the data set provided
        results = analyze_dataset(file_path)
        print("analysis results..", results)

        # llm stage
        llm_response = call_llm(*results)
        

        # README generation stage
       # write_readme(llm_response)

    except Exception as e:
        print(f"Error in autolysis:main: {e}")
    print("autolysis:main completed")

if __name__ == "__main__":
    main()

