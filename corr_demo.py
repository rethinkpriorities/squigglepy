import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

import pandas as pd

import squigglepy as sq

a, b = sq.uniform(-1, 1), sq.to(0, 3)
a, b = sq.correlate((a, b), [[1, 0.9], [0.9, 1]])
a_corr_samples, b_corr_samples = a @ 10_000, b @ 10_000

print(spearmanr(a_corr_samples, b_corr_samples))

a, b = sq.UniformDistribution(-1, 1), sq.to(0, 3)
a_uncorr_samples, b_uncorr_samples = a @ 10_000, b @ 10_000

# Create a dataframe with the correlated samples, using correlated as a categorical variable
df = pd.DataFrame(
    {
        "a": a_corr_samples,
        "b": b_corr_samples,
        "correlated": [True for _ in range(10_000)],
    }
)

print(f"Forced correlation: {df.corr(method='spearman', numeric_only=True)}")
# Append the uncorrelated samples to the dataframe, using correlated as a categorical variable
df2 = pd.DataFrame(
    {
        "a": a_uncorr_samples,
        "b": b_uncorr_samples,
        "correlated": [False for _ in range(10_000)],
    }
)
print(f"Uncorrelated: {df2.corr(method='spearman', numeric_only=True)}")

# Concatenate the two dataframes
df = pd.concat([df, df2])

# Plot the data
sns.jointplot(data=df, x="a", y="b", hue="correlated")
plt.show()
