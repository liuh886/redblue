import pandas as pd
import utm


def apply_kalman_filter(mu,cov):
    # Your Kalman Filter logic goes here
    
    df = pd.DataFrame({'raw': [1,2,3,4,5],
                       'uncertain': [1,2,3,4,5],
                       'filtered': [1,2,3,4,5]})
    
    return df

