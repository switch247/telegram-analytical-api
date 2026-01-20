from pydantic import BaseModel


class TransactionInput(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: int
    TransactionStartTime: str
    PricingStrategy: int


class PredictionOutput(BaseModel):
    is_fraud: bool
    probability: float
    model_version: str
