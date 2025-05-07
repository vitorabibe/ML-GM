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
        self.feature_columns = {}  # Store feature columns for each position group
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
    
    def load_historical_grades(self):
        self.historical_picks = pd.read_csv("Boards (Historical and Present)/draft_class_2015-2020.csv")
        self.historical_qb_grades = pd.read_csv("Historical Grades/QB_grades.csv")
        self.historical_rb_grades = pd.read_csv("Historical Grades/RB_grades.csv")
        self.historical_rec_grades = pd.read_csv("Historical Grades/REC_grades.csv")
        self.historical_ol_grades = pd.read_csv("Historical Grades/OL_grades.csv")
        self.historical_dl_grades = pd.read_csv("Historical Grades/DL_grades.csv")
        self.historical_ed_grades = pd.read_csv("Historical Grades/ED_grades.csv")
        self.historical_lb_grades = pd.read_csv("Historical Grades/LB_grades.csv")
        self.historical_cb_grades = pd.read_csv("Historical Grades/CB_grades.csv")
        self.historical_s_grades = pd.read_csv("Historical Grades/S_grades.csv")

    def load_grades(self):
        self.qb_grades = pd.read_csv("2024 Draft Eligible Player's Grades/QB_grades.csv")
        self.rb_grades = pd.read_csv("2024 Draft Eligible Player's Grades/RB_grades.csv")
        self.rec_grades = pd.read_csv("2024 Draft Eligible Player's Grades/REC_grades.csv")
        self.ol_grades = pd.read_csv("2024 Draft Eligible Player's Grades/OL_grades.csv")
        self.dl_grades = pd.read_csv("2024 Draft Eligible Player's Grades/DL_grades.csv")
        self.ed_grades = pd.read_csv("2024 Draft Eligible Player's Grades/ED_grades.csv")
        self.lb_grades = pd.read_csv("2024 Draft Eligible Player's Grades/LB_grades.csv")
        self.cb_grades = pd.read_csv("2024 Draft Eligible Player's Grades/CB_grades.csv")
        self.s_grades = pd.read_csv("2024 Draft Eligible Player's Grades/S_grades.csv")

    def clean_historical_data(self):
        self.historical_picks = self.historical_picks.dropna(subset=['wAV'])

    def load_data(self):
        self.load_historical_grades()
        self.load_grades()
        self.clean_historical_data()
    
    def get_player_grades_dataframe(self, position_group, is_historical=True):
        """
        Get all player grades for a position group as a complete dataframe
        """
        if is_historical:
            all_grades = {
                'QB': self.historical_qb_grades,
                'RB': self.historical_rb_grades,
                'REC': self.historical_rec_grades,
                'OL': self.historical_ol_grades,
                'DL': pd.concat([self.historical_dl_grades, self.historical_ed_grades], ignore_index=True),
                'LB': self.historical_lb_grades,
                'DB': pd.concat([self.historical_cb_grades, self.historical_s_grades], ignore_index=True)
            }
        else:
            all_grades = {
                'QB': self.qb_grades,
                'RB': self.rb_grades,
                'REC': self.rec_grades,
                'OL': self.ol_grades,
                'DL': pd.concat([self.dl_grades, self.ed_grades], ignore_index=True),
                'LB': self.lb_grades,
                'DB': pd.concat([self.cb_grades, self.s_grades], ignore_index=True)
            }
        
        if position_group in all_grades:
            return all_grades[position_group]
        return None
    
    def get_player_position_group(self, position):
        """
        Get the position group for a given position
        """
        for group, positions in self.position_groups.items():
            if position in positions:
                return group
        return None
    
    def create_features_target_for_position(self, position_group):
        """
        Create features and target for a position group
        """
        # Get all the grades for this position group
        grades_df = self.get_player_grades_dataframe(position_group, is_historical=True)
        
        if grades_df is None or grades_df.empty:
            print(f"No grades found for position group: {position_group}")
            return None, None
        
        # Drop non-feature columns
        feature_cols = grades_df.columns.drop(['player', 'position', 'franchise_id', 'player_id', 'team_name'])
        self.feature_columns[position_group] = feature_cols
        
        # Get all players in this position group from historical picks
        position_players = []
        
        for pos in self.position_groups[position_group]:
            position_players.extend(
                self.historical_picks[self.historical_picks['Pos'] == pos]['Player'].tolist()
            )
        
        # Filter grades for only players in historical picks
        filtered_grades = grades_df[grades_df['player'].isin(position_players)]
        
        if filtered_grades.empty:
            print(f"No matching players found for position group: {position_group}")
            return None, None
        
        # Merge with historical picks to get wAV values
        picks_subset = self.historical_picks[['Player', 'wAV']]
        picks_subset.columns = ['player', 'wAV']
        
        merged_data = pd.merge(filtered_grades, picks_subset, on='player', how='inner')
        merged_data = merged_data.dropna(subset=['wAV'])
        
        if merged_data.empty:
            print(f"No matching data with wAV for position group: {position_group}")
            return None, None
        
        # Extract features and target
        X = merged_data[feature_cols]
        y = merged_data['wAV'].values
        
        return X, y
    
    def train(self):
        for position_group in self.position_groups.keys():
            print(f"Training model for position group: {position_group}")
            
            X, y = self.create_features_target_for_position(position_group)
            
            if X is None or y is None or len(X) < 10:  # Require minimum samples
                print(f"Insufficient data for position group: {position_group}")
                continue
            
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
            
            self.models[position_group] = rfr_cv.best_estimator_
            
            y_pred = rfr_cv.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            self.r2_scores[position_group] = r2
            
            print(f"Model for {position_group} trained with R2 score: {r2:.4f}")

    def predict_player_av(self, player, position):
        # Get position group for this position
        position_group = self.get_player_position_group(position)
        
        if position_group is None or position_group not in self.models:
            print(f"No model available for position: {position} (group: {position_group})")
            return None
        
        grades_df = self.get_player_grades_dataframe(position_group, is_historical=False)
                
        player_row = grades_df[grades_df["player"] == player]
        
        if player_row.empty:
            print(f"Player {player} not found in {position_group} grades")
            return None
        
        # Extract features using the same columns from training
        feature_cols = self.feature_columns[position_group]
        player_features = player_row[feature_cols]
        
        model = self.models[position_group]
        y_pred = model.predict(player_features)
        
        return y_pred[0]
                               
    def predict_board_strength(self, path_to_board):
        """
        Predict wAV for all players in a board
        """
        board = pd.read_csv(path_to_board)
        results = []
        
        for index, row in board.iterrows():
            player = row["Player"]
            position = row["Pos"]
            
            if pd.isna(position):
                continue
                
            player_av = self.predict_player_av(player, position)
            
            if player_av is not None:
                results.append([player, position, player_av])
        
        # Sort by predicted AV
        results.sort(key=lambda x: x[2], reverse=True)
        
        final_board = pd.DataFrame(results, columns=["player", "position", "av"])
        final_board.to_csv("final_board.csv")
        
        return final_board

def main():
    model = AVPredictingModel()
    model.load_data()
    model.train()
    board = model.predict_board_strength(path_to_board="Boards (Historical and Present)/2024_board.csv")
    print(board)
    
main()