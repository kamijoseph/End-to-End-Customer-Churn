
import gradio as gr
from src.predict import Predictor
from src.config import CATEGORICAL_COLS, NUMERICAL_COLS

# Load predictor
predictor = Predictor()

def predict_churn(
    gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
    MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
    DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
    Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
):
    try:
        data = {
            "gender": gender,
            "SeniorCitizen": int(SeniorCitizen),
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": int(tenure),
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": float(MonthlyCharges),
            "TotalCharges": TotalCharges
        }
        
        result = predictor.predict(data)
        
        pred = "Yes" if result["churn_prediction"][0] == 1 else "No"
        prob = result["churn_probability"][0]
        
        return pred, f"{prob:.4f}"
    except Exception as e:
        return "Error", str(e)

# Define Inputs
inputs = [
    gr.Radio(["Female", "Male"], label="Gender"),
    gr.Radio([0, 1], label="Senior Citizen (0=No, 1=Yes)"),
    gr.Radio(["Yes", "No"], label="Partner"),
    gr.Radio(["Yes", "No"], label="Dependents"),
    gr.Slider(0, 72, step=1, label="Tenure (months)"),
    gr.Radio(["Yes", "No"], label="Phone Service"),
    gr.Radio(["No phone service", "No", "Yes"], label="Multiple Lines"),
    gr.Radio(["DSL", "Fiber optic", "No"], label="Internet Service"),
    gr.Radio(["No internet service", "No", "Yes"], label="Online Security"),
    gr.Radio(["No internet service", "No", "Yes"], label="Online Backup"),
    gr.Radio(["No internet service", "No", "Yes"], label="Device Protection"),
    gr.Radio(["No internet service", "No", "Yes"], label="Tech Support"),
    gr.Radio(["No internet service", "No", "Yes"], label="Streaming TV"),
    gr.Radio(["No internet service", "No", "Yes"], label="Streaming Movies"),
    gr.Radio(["Month-to-month", "One year", "Two year"], label="Contract"),
    gr.Radio(["Yes", "No"], label="Paperless Billing"),
    gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method"),
    gr.Number(label="Monthly Charges"),
    gr.Textbox(label="Total Charges (e.g., '100.5' or ' ' for new)")
]

# Define Outputs
outputs = [
    gr.Textbox(label="Churn Prediction (Threshold 0.3)"),
    gr.Textbox(label="Churn Probability")
]

# Create Interface
demo = gr.Interface(
    fn=predict_churn,
    inputs=inputs,
    outputs=outputs,
    title="Customer Churn Prediction System",
    description="Predict customer churn using an XGBoost model. (Threshold = 0.3)",
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
