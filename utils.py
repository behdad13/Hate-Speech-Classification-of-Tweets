import pandas as pd

def balance_classes(data, class_column='class', random_state=42):
    """
    Balance the dataset by undersampling the majority classes.

    Args:
        data (pd.DataFrame): The input dataset with class labels.
        class_column (str): The name of the column containing class labels.
        random_state (int): The random state for reproducibility.

    Returns:
        pd.DataFrame: The balanced dataset.
    """
    # Get the minimum class count
    min_class_count = data[class_column].value_counts().min()

    # Create an empty DataFrame to store the balanced data
    balanced_data = pd.DataFrame(columns=data.columns)

    # Balance the dataset
    for class_value in data[class_column].unique():
        class_data = data[data[class_column] == class_value]
        balanced_class_data = class_data.sample(min_class_count, random_state=random_state)
        balanced_data = balanced_data.append(balanced_class_data, ignore_index=True)

    # Shuffle the balanced dataset
    balanced_data = balanced_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return balanced_data
