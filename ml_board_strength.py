import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

class AVPredictingModel:
    
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.r2_scores = {}
        self.position_groups = {
            'OL': ['T', 'G', 'C', 'OL'],
            'DL': ['DE', 'DT', 'NT', 'DL', 'DI'],
            'LB': ['OLB', 'ILB', 'LB'],
            'DB': ['CB', 'S', 'DB', 'FS', 'SS', 'SAF'],
            'REC': ['WR', 'TE'],
            'QB': ['QB'],
            'RB': ['RB', 'HB', 'FB'],
        }
        self.position_feature_columns = {
            "QB": ["accuracy_percent", "avg_depth_of_target", "btt_rate", "grades_pass", "qb_rating", "yards"],
            "RB": ["grades_offense", "grades_run", "avoided_tackles", "elusive_rating", "yards", "yards_after_contact"],
            "REC": ["grades_offense", "yards", "yprr", "targeted_qb_rating", "yards_after_catch_per_reception", "yards_per_reception"],
            "OL": ["grades_offense", "block_percent", "grades_pass_block", "grades_run_block", "sacks_allowed", "pressures_allowed"],
            "DL": ["grades_defense", "sacks", "total_pressures", "pressures", "tackles_for_loss", "qb_hits"],
            "ED": ["grades_defense", "sacks", "total_pressures", "pressures", "tackles_for_loss", "qb_hits"],
            "LB": ["grades_defense", "grades_run_defense", "sacks", "grades_coverage_defense", "tackles_for_loss", "grades_pass_rush_defense"],
            "CB": ["grades_defense", "grades_coverage_defense", "tackles", "interceptions", "batted_passes"],
            "S": ["grades_defense", "grades_coverage_defense", "tackles", "interceptions", "batted_passes"],
        }

    
    def load_historical_grades(self):
        self.historical_picks = pd.read_csv("Boards (Historical and Present)/draft_class_2015-2020.csv")
        historical_qb_grades = pd.read_csv("Historical Grades/QB_grades.csv")
        self.historical_qb_grades_df = pd.DataFrame(historical_qb_grades, columns=["player","position","accuracy_percent","avg_depth_of_target","btt_rate","grades_pass","qb_rating", "yards"])
        historical_rb_grades = pd.read_csv("Historical Grades/RB_grades.csv")
        self.historical_rb_grades_df = pd.DataFrame(historical_rb_grades, columns=["player","position","grades_offense","grades_run","avoided_tackles","elusive_rating","yards", "yards_after_contact"])
        historical_rec_grades = pd.read_csv("Historical Grades/REC_grades.csv")
        self.historical_rec_grades_df = pd.DataFrame(historical_rec_grades, columns=["player","position","grades_offense","yards","yprr","targeted_qb_rating","yards_after_catch_per_reception","yards_per_reception"])
        historical_ol_grades = pd.read_csv("Historical Grades/OL_grades.csv")
        self.historical_ol_grades_df = pd.DataFrame(historical_ol_grades, columns=["player","position","grades_offense","block_percent","grades_pass_block","grades_run_block","sacks_allowed", "pressures_allowed"])
        historical_dl_grades = pd.read_csv("Historical Grades/DL_grades.csv")
        self.historical_dl_grades_df = pd.DataFrame(historical_dl_grades, columns=["player","position","grades_defense","sacks","total_pressures","pressures","tackles_for_loss","qb_hits"])
        historical_ed_grades = pd.read_csv("Historical Grades/ED_grades.csv")
        self.historical_ed_grades_df = pd.DataFrame(historical_ed_grades, columns=["player","position","grades_defense","sacks","total_pressures","pressures","tackles_for_loss","qb_hits"])
        historical_lb_grades = pd.read_csv("Historical Grades/LB_grades.csv")
        self.historical_lb_grades_df = pd.DataFrame(historical_lb_grades, columns=["player","position","grades_defense","grades_run_defense","sacks","grades_coverage_defense","tackles_for_loss","grades_pass_rush_defense"])
        historical_cb_grades = pd.read_csv("Historical Grades/CB_grades.csv")
        self.historical_cb_grades_df = pd.DataFrame(historical_cb_grades, columns=["player","position","grades_defense","grades_coverage_defense","tackles","interceptions","batted_passes"])
        historical_s_grades = pd.read_csv("Historical Grades/S_grades.csv")
        self.historical_s_grades_df = pd.DataFrame(historical_s_grades, columns=["player","position","grades_defense","grades_coverage_defense","tackles","interceptions","batted_passes"])

    def load_grades(self):
        qb_grades = pd.read_csv("Draft Eligeable Player's Grades/QB_grades.csv")
        self.qb_grades_df = pd.DataFrame(qb_grades, columns=["player","position","accuracy_percent","avg_depth_of_target","btt_rate","grades_pass","qb_rating", "yards"])
        rb_grades = pd.read_csv("Draft Eligeable Player's Grades/RB_grades.csv")
        self.rb_grades_df = pd.DataFrame(rb_grades, columns=["player","position","grades_offense","grades_run","avoided_tackles","elusive_rating","yards", "yards_after_contact"])
        rec_grades = pd.read_csv("Draft Eligeable Player's Grades/REC_grades.csv")
        self.rec_grades_df = pd.DataFrame(rec_grades, columns=["player","position","grades_offense","yards","yprr","targeted_qb_rating","yards_after_catch_per_reception","yards_per_reception"])
        ol_grades = pd.read_csv("Draft Eligeable Player's Grades/OL_grades.csv")
        self.ol_grades_df = pd.DataFrame(ol_grades, columns=["player","position","grades_offense","block_percent","grades_pass_block","grades_run_block","sacks_allowed", "pressures_allowed"])
        dl_grades = pd.read_csv("Draft Eligeable Player's Grades/DL_grades.csv")
        self.dl_grades_df = pd.DataFrame(dl_grades, columns=["player","position","grades_defense","sacks","total_pressures","pressures","tackles_for_loss","qb_hits"])
        ed_grades = pd.read_csv("Draft Eligeable Player's Grades/ED_grades.csv")
        self.ed_grades_df = pd.DataFrame(ed_grades, columns=["player","position","grades_defense","sacks","total_pressures","pressures","tackles_for_loss","qb_hits"])
        lb_grades = pd.read_csv("Draft Eligeable Player's Grades/LB_grades.csv")
        self.lb_grades_df = pd.DataFrame(lb_grades, columns=["player","position","grades_defense","grades_run_defense","sacks","grades_coverage_defense","tackles_for_loss","grades_pass_rush_defense"])
        cb_grades = pd.read_csv("Draft Eligeable Player's Grades/CB_grades.csv")
        self.cb_grades_df = pd.DataFrame(cb_grades, columns=["player","position","grades_defense","grades_coverage_defense","tackles","interceptions","batted_passes"])
        s_grades = pd.read_csv("Draft Eligeable Player's Grades/S_grades.csv")
        self.s_grades_df = pd.DataFrame(s_grades, columns=["player","position","grades_defense","grades_coverage_defense","tackles","interceptions","batted_passes"])

    def clean_historical_data(self):
        self.historical_picks = self.historical_picks.dropna(subset=['wAV'])

    def load_data(self):
        self.load_historical_grades()
        self.load_grades()
        self.clean_historical_data()
    
    def get_player_grades_features(self, player, is_historical=True):
        if is_historical:
            grades = [
                self.historical_qb_grades_df, self.historical_rb_grades_df,
                self.historical_rec_grades_df, self.historical_ol_grades_df,
                self.historical_dl_grades_df, self.historical_ed_grades_df,
                self.historical_lb_grades_df, self.historical_cb_grades_df,
                self.historical_s_grades_df
            ]
        else:
            grades = [
                self.qb_grades_df, self.rb_grades_df, self.rec_grades_df,
                self.ol_grades_df, self.dl_grades_df, self.ed_grades_df,
                self.lb_grades_df, self.cb_grades_df, self.s_grades_df
            ]

        for grade in grades:
            if player in grade["player"].values:
                player_row = grade[grade["player"] == player]
                position = self.get_player_position(player_row["position"].values[0])
                feature_columns = self.position_feature_columns.get(position, [])
                if feature_columns:
                    return player_row[feature_columns].values.flatten().tolist()
        return None

    def create_features_from_board(self, positions=None):
        X_list = []
        y_list = []

        for position in positions:
            for index, row in self.historical_picks.iterrows():
                player = row["Player"]
                player_position = row["Pos"]

                if position and player_position != position:
                    continue

                features = self.get_player_grades_features(player, is_historical=True)
                player_wav = row["wAV"]

                if features != None and pd.notna(player_wav):
                    if all(pd.notna(feature) for feature in features):
                        X_list.append(features)
                        y_list.append(float(player_wav))

        return X_list, y_list
        
    def get_player_position(self, position):
        for position_group in self.position_groups:
            if position in self.position_groups[position_group]:
                return position_group
        return None

    def train(self):
        unique_positions = self.position_groups.keys()
        
        for position in unique_positions:
            print(f"Training model for position: {position}")
            
            X_list, y_list = self.create_features_from_board(positions=self.position_groups[position])
            X = pd.DataFrame(X_list)
            y = y_list
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)
            rfr = RandomForestRegressor(random_state=13)            
            param_grid = {
                "n_estimators": [100, 200, 300], 
                "max_depth": [10, 20, 30], 
                "min_samples_split": [2, 5, 10], 
                "min_samples_leaf": [1, 2, 4]
            }
            
            rfr_cv = GridSearchCV(
                estimator=rfr, 
                param_grid=param_grid, 
                cv=3, 
                scoring="neg_mean_squared_error", 
                n_jobs=-1
            )
            rfr_cv.fit(X_train, y_train)
            
            self.models[position] = rfr_cv.best_estimator_

    def predict_player_av(self, player, position):
        features = self.get_player_grades_features(player, is_historical=False)
        
        position = self.get_player_position(position)
        model = self.models[position]

        player_features = pd.DataFrame([features])

        y_pred = model.predict(player_features)
        return y_pred[0]
                               
    def predict_board_strength(self, path_to_board):
        board = pd.read_csv(path_to_board)
        X_list = []
        
        for index, row in board.iterrows():
            player = row["Player"]
            position = row["Pos"]
            if position == None:
                continue
            player_av = self.predict_player_av(player, position)
            if player_av != None:
                X_list.append([player, position, player_av])
        
        X_list.sort(key=lambda x: x[2], reverse=True)
        final_board = pd.DataFrame(X_list, columns=["player", "position", "av"])
        final_board.to_csv("finalBoard.csv")
        
        return final_board

def main():
    model = AVPredictingModel()
    model.load_data()
    model.train()
    board = model.predict_board_strength(path_to_board="Boards (Historical and Present)/board.csv")
    print(board)
    
main()