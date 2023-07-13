
def format_dataframe(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Creates a Copy of the Original DataFrame. Formats the DataFrame column names to lowercase and formats the specified column values to lowercase
    with underscores instead of spaces.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name: The column to alter
    Returns:
        df (pd.DataFrame): The modified DataFrame with lowercase column names and formatted 'Item' values.
    """

    # Rename columns to lowercase
    df.columns = df.columns.str.lower()

    # Format column values to lowercase with underscores
    df[column_name] = df[column_name].str.lower().str.replace(' ', '_')

    return df

def move_column_to_beginning(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Moves a given column to the beginning of a DataFrame.

    Args:
        - df (pd.DataFrame): The DataFrame to modify.
        - column_name (str): The name of the column to move.

    Returns:
        df (pd.DataFrame): The modified DataFrame with the specified column moved to the beginning.
    """
    # Identify the column name and store it in a variable
    column_to_move = df[column_name]
    
    # Drop the column
    df = df.drop(column_name, axis=1)
    
    # Insert the column in the beginning as index 0 and axis 1
    df.insert(0, column_name, column_to_move)
    
    return df

def drop_values_from_column(df: pd.DataFrame, column: str, values: list) -> pd.DataFrame:
    """
    Drops specified values from a given column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column from which values need to be dropped.
        values (list): A list of values to be dropped from the column.

    Returns:
        df (pd.DataFrame): The modified DataFrame with dropped values from the specified column.
    """

    # Drop rows containing specified values from the column
    df = df[df[column].isin(values) == False].reset_index(drop=True)

    return df

def convert_column_to_float(df: pd.DataFrame, column_name: str, symbol_to_remove: str) -> pd.DataFrame:
    """
    Convert a column in a DataFrame from object to float64 type,
    remove a specified symbol, and substitute ',' with '.' in its values.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the column.
        column_name (str): The name of the column to convert.
        symbol_to_remove (str): The symbol to remove from the column values.
    
    Returns:
        df (pd.DataFrame): The modified DataFrame with the specified column converted to float64 type,
                      the symbol removed from its values, and ',' substituted with '.'.
    """
    
    # Remove specified symbol from the column values in the copied DataFrame
    df[column_name] = df[column_name].str.replace(symbol_to_remove, '')
    
    # Substitute ',' with '.' in the column values in the copied DataFrame
    df[column_name] = df[column_name].str.replace(',', '.')
    
    # Convert the copied column to float64 type in the copied DataFrame
    df[column_name] = df[column_name].astype('float64')
    
    return df
