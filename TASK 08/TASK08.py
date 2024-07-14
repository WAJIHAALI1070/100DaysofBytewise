#Exercise: Determine the probability of a randomly selected iris flower having a petal length greater than a given value.
import seaborn as sns

# Load the Iris dataset from Seaborn
iris = sns.load_dataset('iris')

# Extract petal length data
petal_length = iris['petal_length']

# Specify the threshold petal length
threshold_length = 4.5

# Calculate the probability of petal length > threshold
prob_greater_than_threshold = (petal_length > threshold_length).mean()

# Print the result
print(f"Probability of petal length > {threshold_length}: {prob_greater_than_threshold:.2f}")
