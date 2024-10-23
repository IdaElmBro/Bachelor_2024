import pandas as pd
import numpy as np

# overall split

def custom_train_test_split(df, random_state=None):
    """
    Custom function to split a dataset into training and test sets, ensuring:
    - split based on participants with 6 in each group
    - Test set includes only boys (Gender == 1.0).
    - Stratified by diagnosis (ASD) and participant ID.
    - Varied 'ExpressiveLangRaw1' values in the test set using quantile-based stratification.

    """
    if random_state:
        np.random.seed(random_state)


    # I made the expressivelangraw1 column that contains the values from visit 1
    # so we make sure that each participant only have 1 row (no duplicates across quantiles)
    participant_df = df.groupby('Participant').first().reset_index()    

    # we only want boys in the test
    boys_df = participant_df[(participant_df['Gender'] == 1.0)]

    # make the quantiles
    boys_df['quantile'] = pd.qcut(boys_df['ExpressiveLangRaw1'], q=6, labels=False, duplicates='drop')
    
    test_participants = []

    for asd in [0.0, 1.0]:  # Loop through both groups
        asd_participants = boys_df[boys_df['ASD'] == asd].copy()

        # sample one participant from each quantile to ensure variation
        sampled = asd_participants.groupby('quantile').sample(n=1, random_state=random_state)
        test_participants.extend(sampled['Participant'].tolist()) # add to list

    # match the participants with all their data to get test and train sets
    test_df = df[df['Participant'].isin(test_participants)]
    train_df = df[~df['Participant'].isin(test_participants)]

    # prepare feature sets and labels
    X_train, y_train = train_df.drop(columns=['ASD']), train_df['ASD']
    X_test, y_test = test_df.drop(columns=['ASD']), test_df['ASD']

    return X_train, y_train, X_test, y_test



# same participants on new visits

def train_test_split_by_visit_same(df, random_state=None):
    """
    Splits the whole dataset into six different train-test splits, holding one visit out for testing each time.
    This creates train/test split with the same participants, but different visits (timepoints).
    Returns a dictionary containing X_train, y_train, X_test, and y_test for each visit.
    """
    columns_to_remove = ['Gender', 'Age', 'ExpressiveLangRaw', 'ExpressiveLangRaw1', 'Participant']
    #columns_to_remove = ['ExpressiveLangRaw', 'Participant', 'Age'] # try to include expressivelang, gender and age 

    df = df.drop(columns=columns_to_remove, errors='ignore')
    
    if random_state:
        np.random.seed(random_state)
    
    visits = sorted(df['Visit'].unique())  # get all unique visit numbers

    train_test_splits = {} 

    for test_visit in visits:
        # Train set: all visits except the current test visit
        train_df = df[df['Visit'] != test_visit]
        # Test set: only the current test visit
        test_df = df[df['Visit'] == test_visit]

        # drop the quantile column as well 
        X_train, y_train = train_df.drop(columns=['ASD']), train_df['ASD']
        X_test, y_test = test_df.drop(columns=['ASD']), test_df['ASD']

        train_test_splits[f'visit_{test_visit}'] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }

    return train_test_splits


# new participants on new visits

def train_test_split_by_visit_new(X_train, y_train, X_test, y_test, visits):
    """
    Split the existing training and test sets into separate train-test splits,
    holding one visit out for testing each time, ensuring no participant or visit overlap. 
    This function stores the new splits in dictionaries.
    """

    train_test_splits = {}  
    
    for test_visit in visits: # loop through each visit
        test_mask = X_test['Visit'] == test_visit
        
        # test set for the current visit
        X_test_visit = X_test[test_mask]
        y_test_visit = y_test[test_mask]

        # train set excluding the current test visit
        X_train_visit = X_train[X_train['Visit'] != test_visit]
        y_train_visit = y_train[X_train['Visit'] != test_visit]

        train_test_splits[f'visit_{test_visit}'] = {
            'X_train': X_train_visit,
            'y_train': y_train_visit,
            'X_test': X_test_visit,
            'y_test': y_test_visit
        }
    return train_test_splits