import streamlit as st # pip install streamlit
from sklearn.ensemble import RandomForestRegressor
from explainerdashboard import RegressionExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_fare, feature_descriptions

X_train, y_train, X_test, y_test = titanic_fare()
model = RandomForestRegressor(n_estimators=50, max_depth=10).fit(X_train, y_train)

explainer = RegressionExplainer(model, X_test, y_test, 
                                cats=['Sex', 'Deck', 'Embarked'], 
                                descriptions=feature_descriptions,
                                units="$")

ExplainerDashboard(explainer).run()