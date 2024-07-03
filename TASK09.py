#Exercise: Perform a hypothesis test to determine if there is a significant difference in the mean petal length between two species of iris flowers.
import seaborn as sns
from scipy.stats import ttest_ind

# Load the Iris dataset from Seaborn
iris = sns.load_dataset('iris')

# Extract petal lengths for two species (e.g., setosa and versicolor)
petal_length_setosa = iris[iris['species'] == 'setosa']['petal_length']
petal_length_versicolor = iris[iris['species'] == 'versicolor']['petal_length']

# Perform independent t-test
t_statistic, p_value = ttest_ind(petal_length_setosa, petal_length_versicolor)

# Print the results
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the p-value
alpha = 0.05
if p_value < alpha:
    print("There is a significant difference in the mean petal length between setosa and versicolor.")
else:
    print("There is no significant difference in the mean petal length between setosa and versicolor.")
