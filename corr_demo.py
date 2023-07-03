from scipy.stats import spearmanr

import pandas as pd

import squigglepy as sq

a, b = sq.beta(20, 8), sq.exponential(1)
a, b = sq.correlate((a, b), [[1, 0.5], [0.5, 1]])
a_corr_samples, b_corr_samples = a @ 10_000, b @ 10_000

print(spearmanr(a_corr_samples, b_corr_samples))

a, b = sq.beta(1, 0.025), sq.exponential(1)
a_uncorr_samples, b_uncorr_samples = a @ 10_000, b @ 10_000

# Create a dataframe with the correlated samples, using correlated as a categorical variable
corr = pd.DataFrame(
    {
        "a": a_corr_samples,
        "b": b_corr_samples,
        "correlated": [True for _ in range(10_000)],
    }
)

print(f"Forced correlation: {corr.corr(method='spearman', numeric_only=True)}")
# Append the uncorrelated samples to the dataframe, using correlated as a categorical variable
uncorr = pd.DataFrame(
    {
        "a": a_uncorr_samples,
        "b": b_uncorr_samples,
        "correlated": [False for _ in range(10_000)],
    }
)
print(f"Uncorrelated: {uncorr.corr(method='spearman', numeric_only=True)}")

# Concatenate the two dataframes
df = pd.concat([uncorr, corr])

# Plot the data
# sns.jointplot(data=uncorr, x="a", y="b")
# plt.show()
# sns.jointplot(data=corr, x="a", y="b")
# # sns.jointplot(data=df, x="a", y="b", hue="correlated")
# plt.show()
