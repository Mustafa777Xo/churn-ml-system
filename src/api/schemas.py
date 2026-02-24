from typing import List, Optional, Union
from pydantic import BaseModel, model_validator


class CustomerInput(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Union[float, str]


class PredictRequest(BaseModel):
    record: Optional[CustomerInput] = None
    records: Optional[List[CustomerInput]] = None

    @model_validator(mode="after")
    def check_one_of(self):
        if (self.record is None) == (self.records is None):
            raise ValueError("Provide exactly one of 'record' or 'records'")
        return self


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: int
    threshold: float
    model_version: str
