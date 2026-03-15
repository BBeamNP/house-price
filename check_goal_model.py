import joblib

model = joblib.load("app/ml_models/goal_minemap_model.pkl")

print("Model type:", type(model))

print("Number of neighbors:", model.n_neighbors)

print("Algorithm:", model.algorithm)

print("Metric:", model.metric)