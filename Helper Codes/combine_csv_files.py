#combines all the QB_grades into one csv file

import pandas as pd

QB_Grades = pd.DataFrame()
RB_Grades = pd.DataFrame()
REC_Grades = pd.DataFrame()
OL_Grades = pd.DataFrame()
DL_Grades = pd.DataFrame()
ED_Grades = pd.DataFrame()
LB_Grades = pd.DataFrame()
CB_Grades = pd.DataFrame()
S_Grades = pd.DataFrame()

for year in range(2023, 2024):
    QB_grades = pd.concat([QB_Grades, pd.read_csv(f"QB_grades {year}.csv")])
    RB_grades = pd.concat([RB_Grades, pd.read_csv(f"RB_grades {year}.csv")])
    REC_grades = pd.concat([REC_Grades, pd.read_csv(f"REC_grades {year}.csv")])
    OL_grades = pd.concat([OL_Grades, pd.read_csv(f"OL_grades {year}.csv")])
    DL_grades = pd.concat([DL_Grades, pd.read_csv(f"DL_grades {year}.csv")])
    ED_grades = pd.concat([ED_Grades, pd.read_csv(f"ED_grades {year}.csv")])
    LB_grades = pd.concat([LB_Grades, pd.read_csv(f"LB_grades {year}.csv")])
    CB_grades = pd.concat([CB_Grades, pd.read_csv(f"CB_grades {year}.csv")])
    S_grades = pd.concat([S_Grades, pd.read_csv(f"S_grades {year}.csv")])

QB_grades.to_csv("QB_grades.csv", index=False)
RB_grades.to_csv("RB_grades.csv", index=False)
REC_grades.to_csv("REC_grades.csv", index=False)
OL_grades.to_csv("OL_grades.csv", index=False)
DL_grades.to_csv("DL_grades.csv", index=False)
ED_grades.to_csv("ED_grades.csv", index=False)
LB_grades.to_csv("LB_grades.csv", index=False)
CB_grades.to_csv("CB_grades.csv", index=False)
S_grades.to_csv("S_grades.csv", index=False)