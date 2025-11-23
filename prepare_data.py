import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = 'data/raw_data.csv'
TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'
TARGET_COL = 'cpu_usage'
FEATURES = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']

def prepare_data():
    print("Starting data preparation...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Data Cleaning and Feature Engineering
    df = df[FEATURES + [TARGET_COL]]
    df.dropna(inplace=True)
    df = pd.get_dummies(df, columns=['controller_kind'], drop_first=True)
    
    # Split and save processed data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    
    print(f"Data split and saved: Train ({len(train_df)}), Test ({len(test_df)})")

if __name__ == "__main__":
    prepare_data()