import sys
from pandas import DataFrame, read_csv
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = read_csv(messages_filepath)
    categories = read_csv(categories_filepath)
    return messages, categories


def categories_return_values(row):
    """cleaning function to be applied to the categories column"""
    values = []
    for column in row:
        column_name, value = column.split('-')
        values.append(value)
    return values


def categories_get_column_names(row):
    """gets column names from the original dataset"""
    column_names = []
    for column in row:
        column_name, value = column.split('-')
        column_names.append(column_name)
    return column_names


def clean_data_categories(df_categories):
    column_names = categories_get_column_names(df_categories['categories'][0].split(';'))
    temporary_df = df_categories['categories'].str.split(';', expand=True).apply(categories_return_values)
    temporary_df.columns = column_names
    temporary_df['id'] = df_categories['id']
    return temporary_df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df_messages, df_categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning category data...')
        df_categories = clean_data_categories(df_categories)

        print('Merging data...')
        df = df_messages.merge(df_categories, how='left', on='id')

        print('Eliminating duplicates...')
        df.drop_duplicates(inplace=True)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
