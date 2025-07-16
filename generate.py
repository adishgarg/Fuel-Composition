import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
import matplotlib.pyplot as plt
import seaborn as sns
from sdv.metadata import SingleTableMetadata

df = pd.read_csv('fuel_data.csv')

# print(df.head())
# print(df.describe())

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

gc_model = GaussianCopulaSynthesizer(metadata=metadata)
gc_model.fit(df)

ctgan_model = CTGANSynthesizer(metadata=metadata, epochs=1000, batch_size=500, )
ctgan_model.fit(df)

synthetic_gc = gc_model.sample(5000)
synthetic_ctgan = ctgan_model.sample(5000)

synthetic_gc.to_csv('synthetic_fuel_data_gc.csv', index=False)
synthetic_ctgan.to_csv('synthetic_fuel_data_ctgan.csv', index=False)

print("Synthetic datasets saved ✅")

for col in ['frequency_shift', 'S11_magnitude', 'Q_factor', 'phase_shift']:
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df[col], label='Original', color='blue')
    sns.kdeplot(synthetic_gc[col], label='GaussianCopula', color='green')
    sns.kdeplot(synthetic_ctgan[col], label='CTGAN', color='orange')
    plt.title(f'Distribution Comparison for {col}')
    plt.legend()
    plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(y='label', data=df, color='blue', label='Original')
sns.countplot(y='label', data=synthetic_gc, color='green', alpha=0.5, label='GaussianCopula')
plt.title('Label Distribution Comparison')
plt.legend()
plt.show()