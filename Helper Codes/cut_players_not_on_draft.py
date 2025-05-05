import pandas as pd

QB_grades = pd.read_csv("QB_grades.csv")
RB_grades = pd.read_csv("RB_grades.csv")
REC_grades = pd.read_csv("REC_grades.csv")
OL_grades = pd.read_csv("OL_grades.csv")
DL_grades = pd.read_csv("DL_grades.csv")
ED_grades = pd.read_csv("ED_grades.csv")
LB_grades = pd.read_csv("LB_grades.csv")
CB_grades = pd.read_csv("CB_grades.csv")
S_grades = pd.read_csv("S_grades.csv")
board = pd.read_csv("Boards (Historical and Present)/board.csv")
# Deletes player's grades who are not orginillay from that position
QB_grades = QB_grades[QB_grades["position"] == "QB"]
RB_grades = RB_grades[(RB_grades["position"] == "HB") | (RB_grades["position"] == "FB")]
REC_grades = REC_grades[(REC_grades["position"] == "WR") | (REC_grades["position"] == "TE")]
OL_grades = OL_grades[(OL_grades["position"] == "T") | (OL_grades["position"] == "G") | (OL_grades["position"] == "C")]
DL_grades = DL_grades[DL_grades["position"] == "DI"]
ED_grades = ED_grades[ED_grades["position"] == "ED"]
LB_grades = LB_grades[LB_grades["position"] == "LB"]
CB_grades = CB_grades[CB_grades["position"] == "CB"]
S_grades = S_grades[S_grades["position"] == "S"]

# Deletes players who are not in the draft_class
QB_grades = QB_grades[QB_grades["player"].isin(board["Player"])]
RB_grades = RB_grades[RB_grades["player"].isin(board["Player"])]
REC_grades = REC_grades[REC_grades["player"].isin(board["Player"])]
OL_grades = OL_grades[OL_grades["player"].isin(board["Player"])]
DL_grades = DL_grades[DL_grades["player"].isin(board["Player"])]
ED_grades = ED_grades[ED_grades["player"].isin(board["Player"])]
LB_grades = LB_grades[LB_grades["player"].isin(board["Player"])]
CB_grades = CB_grades[CB_grades["player"].isin(board["Player"])]
S_grades = S_grades[S_grades["player"].isin(board["Player"])]

#creates new csv file only with the remaining players
QB_grades.to_csv("Draft Eligeable Player's Grades/QB_grades.csv", index=False)
RB_grades.to_csv("Draft Eligeable Player's Grades/RB_grades.csv", index=False)
REC_grades.to_csv("Draft Eligeable Player's Grades/REC_grades.csv", index=False)
OL_grades.to_csv("Draft Eligeable Player's Grades/OL_grades.csv", index=False)
DL_grades.to_csv("Draft Eligeable Player's Grades/DL_grades.csv", index=False)
ED_grades.to_csv("Draft Eligeable Player's Grades/ED_grades.csv", index=False)
LB_grades.to_csv("Draft Eligeable Player's Grades/LB_grades.csv", index=False)
CB_grades.to_csv("Draft Eligeable Player's Grades/CB_grades.csv", index=False)
S_grades.to_csv("Draft Eligeable Player's Grades/S_grades.csv", index=False)