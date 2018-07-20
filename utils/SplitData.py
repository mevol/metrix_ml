class SplitData():
    def self():
        pass
    
    def split_data(df):
        X = df
        y = df['EP_success']
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_work = X_train.copy()
        return X_train_work, X_train

    #split_data(metrix)