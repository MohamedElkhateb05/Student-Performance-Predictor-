# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error ,mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df=pd.read_csv("/kaggle/input/student-performance-factors/StudentPerformanceFactors.csv")
df2=df
df.head()

df.describe()

df.info()


print(f"Dataset shape: {df.shape}")
#print(f"Missing values:\n{df.isnull().sum()}")

import missingno as msno
msno.matrix(df)  

(df.isnull().sum() / len(df)) * 100

categorical_missing = ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']

fill_dict = {
    'Teacher_Quality': 'Missing',
    'Parental_Education_Level': 'Missing',
    'Distance_from_Home': 'Missing'
}

df.fillna(value=fill_dict, inplace=True)


print("\nUnique values in 'Teacher_Quality' after imputation:")
print(df['Teacher_Quality'].unique())
print("\nUnique values in 'Parental_Education_Level after imputation:")
print(df['Parental_Education_Level'].unique())
print("\nUnique values in 'Distance_from_Home' after imputation:")
print(df['Distance_from_Home'].unique())


df.isnull().sum()


import plotly.express as px
from ipywidgets import interact, Dropdown


print("DataFrame dtypes:")
print(df.dtypes)
print("\n" + "="*50)


categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)


numerical_features = numerical_cols  # Don't remove Exam_Score!

default_numerical = 'Exam_Score' if 'Exam_Score' in numerical_cols else numerical_cols[0]

def create_interactive_plot(categorical_feature, numerical_feature, plot_type):
    """
    Creates an interactive plot based on user selection
    """
    if plot_type == 'Box Plot':
        fig = px.box(df, x=categorical_feature, y=numerical_feature, 
                     title=f'Distribution of {numerical_feature} by {categorical_feature}')
    elif plot_type == 'Bar Chart (Averages)':
        # Calculate averages for each category
        avg_data = df.groupby(categorical_feature)[numerical_feature].mean().reset_index()
        fig = px.bar(avg_data, x=categorical_feature, y=numerical_feature,
                     title=f'Average {numerical_feature} by {categorical_feature}')
    elif plot_type == 'Violin Plot':
        fig = px.violin(df, x=categorical_feature, y=numerical_feature,
                       title=f'Distribution of {numerical_feature} by {categorical_feature}')
    
    # Update layout for better readability
    fig.update_layout(
        width=800,
        height=500,
        font=dict(size=12)
    )
    
    fig.show()
  
# 4. Create the interactive widget
@interact
def explore_data(
    categorical_feature=Dropdown(
        options=categorical_cols,
        value=categorical_cols[0] if categorical_cols else None,  # Safe default
        description='Categorical Feature:',
        style={'description_width': 'initial'}
    ),
    numerical_feature=Dropdown(
        options=numerical_features,
        value=default_numerical,  # Safe default
        description='Numerical Feature:',
        style={'description_width': 'initial'}
    ),
    plot_type=Dropdown(
        options=['Box Plot', 'Bar Chart (Averages)', 'Violin Plot'],
        value='Box Plot',  # Default value
        description='Plot Type:',
        style={'description_width': 'initial'}
    )
):
    """
    Main interactive function that creates the dashboard
    """
    create_interactive_plot(categorical_feature, numerical_feature, plot_type)
    print(f"Showing {plot_type} of {numerical_feature} by {categorical_feature}")


fig = px.histogram(df, x='Exam_Score', 
                   title='Distribution of Student Exam Scores',
                   labels={'Exam_Score': 'Final Exam Score'}, 
                   nbins=20) # You can adjust the number of bins

# Add a vertical line for the mean score
mean_score = df['Exam_Score'].mean()
fig.add_vline(x=mean_score, line_dash="dash", line_color="red", 
              annotation_text=f'Mean Score: {mean_score:.2f}', 
              annotation_position="top right")

# Update layout
fig.update_layout(
    xaxis_title="Exam Score",
    yaxis_title="Number of Students",
    showlegend=False
)

fig.show()


print(df['Exam_Score'].describe())


correlation_matrix = df.select_dtypes(include=[np.number]).corr()


fig = px.imshow(correlation_matrix,
                text_auto=True, # Automatically puts the correlation number on each cell
                aspect="auto", # Makes the plot square
                color_continuous_scale='RdBu_r', # Blue (negative) - White (0) - Red (Positive)
                title='<b>Correlation Matrix: Relationship Between All Numerical Features</b>'
               )


fig.update_layout(width=700, height=700,
                 xaxis_title='Features',
                 yaxis_title='Features')

fig.show()


print(" Correlation with Exam_Score (from most positive to most negative):")
target_correlations = correlation_matrix['Exam_Score'].drop('Exam_Score').sort_values(ascending=False)
print(target_correlations)

df_encoded = df.copy()

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

binary_cols = []
multi_category_cols = []

for col in categorical_cols:
    unique_vals = df[col].nunique()
    #print(f"'{col}': {unique_vals} unique values -> {df[col].unique()}")
    if unique_vals == 2:
        binary_cols.append(col)
    else:
        multi_category_cols.append(col)


for col in binary_cols:
    unique_vals = sorted(df[col].unique())
    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
    df_encoded[col] = df_encoded[col].map(mapping)
    #print(f"Label encoded '{col}': {mapping}")


df_encoded = pd.get_dummies(df_encoded, columns=multi_category_cols, prefix=multi_category_cols)


bool_columns = df_encoded.select_dtypes(include=['bool']).columns

df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)

if binary_cols:
    print(f"\nSample of label encoded binary columns ({binary_cols}):")
    print(df_encoded[binary_cols].head())
if multi_category_cols:
    sample_col = multi_category_cols[0]
    encoded_cols = [col for col in df_encoded.columns if sample_col in col]
    print(f"\nSample of one-hot encoded columns for '{sample_col}' (now as 1/0):")
    print(df_encoded[encoded_cols].head())
  
X = df_encoded.drop('Exam_Score', axis=1) 
y = df_encoded['Exam_Score']               


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False) # squared=False returns RMSE
r2 = r2_score(y_test, y_pred)

print("\n MODEL PERFORMANCE METRICS:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")




plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Actual vs Predicted Exam Scores\n(Linear Regression Model)')


plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.2f}', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.show()


