
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def market_basket_analysis(df: pd.DataFrame, transactions: str, items: str, min_support=0.1, metric="lift", min_threshold=1) -> tuple:
    """
    Apply Market Basket Analysis using the Apriori algorithm to a dataframe with transactions and items sold.

    Args:
        df (pd.DataFrame): Input dataframe.
        transactions (str): Name of the first column for analysis, the one with the transactions values.
        items (str): Name of the second column for analysis, the one with the items values.
        min_support (float): Minimum support threshold for frequent itemsets.
        metric (str): Evaluation metric for association rules.
        min_threshold (float): Minimum threshold for the evaluation metric.

    Returns:
        tuple: Tuple containing DataFrame of association rules and DataFrame of frequent itemsets.

    """
    # Group the DataFrame by "transactions" and "item" and calculate the count of each transaction
    grouped_df = df.groupby([transactions, items]).size().reset_index(name="count")

    # Pivot the DataFrame to have "transactions" as index and "items" as columns and sum the counts
    pivot_df = grouped_df.pivot_table(index=transactions, columns=items, values="count", aggfunc="sum", fill_value=0)

    # Encode the values to 1 if greater than 0 and 0 if less than or equal to 0
    def encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    basket_sets = pivot_df.applymap(encode_units)

    # Apply the Apriori algorithm to find frequent itemsets
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

    # Add a column with the length of the itemsets
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

    # Generate association rules based on frequent itemsets
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    # Sort the rules by lift
    rules = rules.sort_values(by='lift', ignore_index=True)

    return rules, frequent_itemsets
