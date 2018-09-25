from model_building import model_LR, model_NB
from preprocessing import X_test
import pandas as pd
import os

templet = pd.read_csv('J:/Code/kaggle/Bags_of_Words_Meets_Bags_of_Popcorn/sampleSubmission.csv')

## NB test and save
templet_column = templet.columns.values.tolist() # 

NB_prediction_test = model_NB.predict(X_test)

NB_prediction_test_df = pd.DataFrame(NB_prediction_test, columns = templet_column[1:])

NB_result = pd.concat([templet['id'], NB_prediction_test_df], axis = 1)

NB_result.to_csv("J:/Code/kaggle/Bags_of_Words_Meets_Bags_of_Popcorn/Result_NB.csv", index = False)

## LR test and save

LR_prediction_test = model_LR.predict(X_test)

LR_prediction_test_df = pd.DataFrame(LR_prediction_test, columns = templet_column[1:])

LR_result = pd.concat([templet['id'], LR_prediction_test_df], axis = 1)

# print(LR_result)

LR_result.to_csv("J:/Code/kaggle/Bags_of_Words_Meets_Bags_of_Popcorn/Result_LR.csv", index = False)