import pandas as pd

def load_collection_as_df(db, collection_name):
    data = list(db[collection_name].find({}, {"_id": 0}))
    return pd.DataFrame(data)
