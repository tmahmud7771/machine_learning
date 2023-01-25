import pandas as pd

data = pd.read_csv('..\..\data_sets\lsd_math_score_data.csv')

time  = data[["Time_Delay_in_Minutes"]]
score = data[["Avg_Math_Test_Score"]]
lsd   = data[["LSD_ppm"]]