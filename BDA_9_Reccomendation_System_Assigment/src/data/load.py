import os
import pandas as pd

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


# compute project root (three levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ML100K_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ml-100k")

GENRES = ["unknown","Action","Adventure","Animation","Children's","Comedy","Crime","Documentary",
          "Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller",
          "War","Western"]

def load_movielens_100k(data_dir=ML100K_DIR):
    udata_path = os.path.join(data_dir, "u.data")
    uitem_path = os.path.join(data_dir, "u.item")
    if not os.path.exists(udata_path) or not os.path.exists(uitem_path):
        raise FileNotFoundError(f"Place ml-100k files under {data_dir}. See README for download link.")
    udata = pd.read_csv(udata_path, sep='\t', names=["user_id","item_id","rating","timestamp"])
    col_names = ["movie_id","title","release_date","video_release_date","imdb_url"] + GENRES
    items = pd.read_csv(uitem_path, sep='|', encoding='latin-1', names=col_names, index_col=False)
    return udata, items