def load_data_from_csv(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def load_data_from_database(connection_string, query):
    import pandas as pd
    from sqlalchemy import create_engine

    engine = create_engine(connection_string)
    return pd.read_sql(query, engine)

def load_raw_data(file_name):
    import os
    raw_data_path = os.path.join('data', 'raw', file_name)
    return load_data_from_csv(raw_data_path)

def load_processed_data(file_name):
    import os
    processed_data_path = os.path.join('data', 'processed', file_name)
    return load_data_from_csv(processed_data_path)