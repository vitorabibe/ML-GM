import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

class NFLDraftTradeModel:
    
    def load_data(self, historical_trades_path):
        # Reads files, then removes quotation marks on lists required for file-reading
        self.historical_trades = pd.read_csv(historical_trades_path)
        self.historical_trades["picks_given"] = self.historical_trades["picks_given"].apply(ast.literal_eval)
        self.historical_trades["picks_received"] = self.historical_trades["picks_received"].apply(ast.literal_eval)

    def calculate_performance_value(self, pick_number):
        '''
        Weibull distribution formula from Massey & Thaler (2013)
        Formula: v(t) = e^(-λ(t-1)^β)
        Value for the first pick is normalized to 1.0
        Based on their analysis of actual NFL draft-day trades
        '''

        lambda_param = 0.146
        beta_param = 0.698

        return np.exp(-lambda_param * ((pick_number - 1) ** beta_param))
    
    def transform_next_years_pick(self, pick, next_year_picks_estimation):
        '''
        This currently can be passed as a dictionary based on predictions on where the other team's pick will be
        next year, as well as based on how much a team values a future pick. I plan on making another model that
        predicts a team's next season record based on their player's grades, their previous records, and other data
        '''
        
        if "1st" in pick:
            return next_year_picks_estimation["Next Year 1st"]
        elif "2nd" in pick:
            return next_year_picks_estimation["Next Year 2nd"]
        elif "3rd" in pick:
            return next_year_picks_estimation["Next Year 3rd"]
        elif "4th" in pick:
            return next_year_picks_estimation["Next Year 4th"]
        elif "5th" in pick:
            return next_year_picks_estimation["Next Year 5th"]
        elif "6th" in pick:
            return next_year_picks_estimation["Next Year 6th"]
        elif "7th" in pick:
            return next_year_picks_estimation["Next Year 7th"]
        
    
    def create_features_from_picks(self, picks_given, picks_received, next_year_picks_estimation):
        total_picks_given_value = 0
        total_picks_received_value = 0
        len_picks_given, len_picks_received = len(picks_given), len(picks_received)
        future_picks_given = []
        future_picks_received = []

        # Adds upp all picks into one value using the Weibull distribution performance value equation
        for pick in picks_given:
            if isinstance(pick, str):
                future_picks_given.append(pick)
                pick = self.transform_next_years_pick(pick, next_year_picks_estimation)
            total_picks_given_value += self.calculate_performance_value(pick)
        for pick in picks_received:
            if isinstance(pick, str):
                future_picks_received.append(pick)
                pick = self.transform_next_years_pick(pick, next_year_picks_estimation)
            total_picks_received_value += self.calculate_performance_value(pick)

        # Deletes future picks without messing up with for loop
        for future_pick in future_picks_given:
            picks_given.remove(future_pick)
        for future_pick in future_picks_received:
            picks_received.remove(future_pick)

        return (
            [
            float(total_picks_given_value), float(total_picks_received_value),
            min(picks_given), min(picks_received)
            ])

    def create_pick_matrix(self, next_year_picks_estimation, picks_given=None, picks_received=None):
        '''
        Edge case created for when the function is called for the proposed trade, 
        keeping all calculations on the same function
        '''

        if picks_given and picks_received:
            features = self.create_features_from_picks(picks_given, picks_received, next_year_picks_estimation)
            matrix = pd.DataFrame([features], columns=[
            "total_value_given", "total_value_received",
            "best_pick_given", "best_pick_received"
        ])
            return matrix
        
        # Creates data frame for test's X
        self.historical_trades = self.historical_trades.drop(columns = ["team_giving_picks", "team_receiving_picks", "year"])
        X_list = []
        for idx, row in self.historical_trades.iterrows():
            features = self.create_features_from_picks(row["picks_given"], row["picks_received"], next_year_picks_estimation)
            X_list.append(features)
            
        self.X = pd.DataFrame(X_list, columns=[
            "total_value_given", "total_value_received",
            "best_pick_given", "best_pick_received"
        ])
        # Creates test's y
        self.y = self.historical_trades["av_difference"]

    def train_and_predict(self, picks_given, picks_received, next_year_picks_estimation):
        self.create_pick_matrix(next_year_picks_estimation)
        proposed_trade_features = self.create_pick_matrix(next_year_picks_estimation, picks_given, picks_received)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=17, test_size=0.2)
        
        # Dictionary with only Decision Tree model and its configurations
        params = {
            "max_depth": [5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
        
        print("Training Decision Tree model...")
        model = DecisionTreeRegressor(random_state=13)
        
        # Tune hyperparameters
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=params,
            cv=3,
            scoring="r2",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # Predict using the best model
        y_pred_new_trade = best_model.predict(proposed_trade_features)
        
        # Print results
        print(f"\nBest Model: Decision Tree")
        print(f"Best R²: {r2:.2f}")
        print(f"Best Parameters: {best_params}")
        print(f"Predicted AV Difference: {y_pred_new_trade[0]:.2f}")
            
        return y_pred_new_trade[0], "Decision Tree", r2

def main():

    model = NFLDraftTradeModel()

    model.load_data(
        historical_trades_path="historical_trades.csv"
    )

    ''' 
    This can be changed based on the user's preferences, prediction of the other team's record, 
    and value given to future pick
    '''
    next_year_picks_estimation = {
                                "Next Year 1st": 20, "Next Year 2nd": 50, 
                                "Next Year 3rd": 85, "Next Year 4th": 120, 
                                "Next Year 5th": 150, "Next Year 6th": 180, 
                                "Next Year 7th": 210
                                  }

    model.train_and_predict([5, "Next Year 1st", 36, 104], [2, 104, 200], next_year_picks_estimation)


main()