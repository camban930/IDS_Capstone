from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def train_random_forest():
    # Load the logs.csv file
    df = pd.read_csv("logs.csv")

    # Feature Engineering: Convert categorical data into numeric and create new features
    df["failed_attempt"] = df["status"].apply(lambda x: 1 if x == "failure" else 0)
    df["success_attempt"] = df["status"].apply(lambda x: 1 if x == "success" else 0)

    # Group by IP to track login patterns
    ip_stats = df.groupby("source_ip").agg({
        "failed_attempt": "sum", 
        "success_attempt": "sum"
    }).reset_index()
    ip_stats["total_attempts"] = ip_stats["failed_attempt"] + ip_stats["success_attempt"]

    # Define labels: Suspicious (1) if failed attempts > 3, else Normal (0)
    ip_stats["suspicious"] = ip_stats["failed_attempt"].apply(lambda x: 1 if x > 3 else 0)

    # Train-test split
    X = ip_stats[["failed_attempt", "success_attempt", "total_attempts"]]
    y = ip_stats["suspicious"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # Save the model for future predictions
    joblib.dump(model, "login_detector.pkl")

    return model  # Return the trained model

def analyze_login_attempts(model):
    """
    Analyze login attempts and predict whether they are suspicious.

    Parameters:
    model: Trained Random Forest model.

    Returns:
    dict: Analysis results for login attempts.
    """

    # Example input data: Replace this with real input if needed
    login_data = pd.DataFrame([
        {"failed_attempt": 5, "success_attempt": 2, "total_attempts": 7},
        {"failed_attempt": 1, "success_attempt": 3, "total_attempts": 4},
        {"failed_attempt": 10, "success_attempt": 0, "total_attempts": 10},
    ])

    # Make predictions
    predictions = model.predict(login_data)
    prediction_proba = model.predict_proba(login_data)

    # Format results
    results = []
    for i, row in login_data.iterrows():
        results.append({
            "failed_attempt": row["failed_attempt"],
            "success_attempt": row["success_attempt"],
            "total_attempts": row["total_attempts"],
            "is_suspicious": bool(predictions[i]),
            "probability": {
                "normal": round(prediction_proba[i][0] * 100, 2),
                "suspicious": round(prediction_proba[i][1] * 100, 2),
            },
        })

    return {"analysis_results": results}

def visualize_comparison(rf_metrics, nb_metrics):
    """
    Visualize the comparison of model metrics.

    Parameters:
    rf_metrics (dict): Metrics for the Random Forest model (precision, recall, F1, accuracy).
    nb_metrics (dict): Metrics for the Naive Bayes model (precision, recall, F1, accuracy).
    """

    # Create lists of metric names and values
    metrics = ["Precision", "Recall", "F1 Score", "Accuracy"]
    rf_values = [rf_metrics["precision"], rf_metrics["recall"], rf_metrics["f1"], rf_metrics["accuracy"]]
    nb_values = [nb_metrics["precision"], nb_metrics["recall"], nb_metrics["f1"], nb_metrics["accuracy"]]

    # Create the bar plot
    x = range(len(metrics))
    width = 0.35  # Width of bars

    fig, ax = plt.subplots()
    ax.bar(x, rf_values, width, label="Random Forest", color="blue")
    ax.bar([i + width for i in x], nb_values, width, label="Naive Bayes", color="orange")

    # Add labels, title, and legend
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Scores")
    ax.set_title("Model Metrics Comparison")
    ax.legend()

    plt.tight_layout()
    plt.show()
