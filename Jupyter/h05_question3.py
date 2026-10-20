import pickle

# Load the trained model
with open("pipeline.bin", "rb") as f:
    model = pickle.load(f)

# Lead data from Question 3
lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41
}

# Predict probability of conversion
probability = model.predict_proba([lead])[0, 1]

print(f"Conversion probability: {probability:.3f}")
