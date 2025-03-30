import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(18, 65, n_samples),
    'sex': np.random.choice(['male', 'female'], n_samples),
    'bmi': np.random.normal(26, 4, n_samples),
    'children': np.random.randint(0, 5, n_samples),
    'smoker': np.random.choice(['yes', 'no'], n_samples),
    'region': np.random.choice(['southeast', 'southwest', 'northeast', 'northwest'], n_samples),
    'charges': np.random.normal(13000, 5000, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('insurance.csv', index=False)
print("Sample dataset created successfully!")