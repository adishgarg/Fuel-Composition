import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('fuel_data.csv')

if 'label_encoded' not in df.columns:
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

fig = px.scatter_3d(
    df,
    x='frequency_shift',
    y='Q_factor',
    z='phase_shift',
    color='label',
    title='3D Feature Distribution by Fuel Type',
    opacity=0.7,
    size_max=10
)
fig.show()

sns.pairplot(df, hue='label', vars=['frequency_shift', 'Q_factor', 'phase_shift', 'S11_magnitude'], palette='husl')
plt.suptitle("2D Feature Distributions by Label", y=1.02)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()