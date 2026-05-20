import torch
import pandas as pd


def predict_customer(
    customer_data,
    model,
    scaler,
    feature_columns,
    threshold=0.6
):

    customer_df = pd.DataFrame(
        [customer_data]
    )

    customer_df = customer_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    customer_scaled = scaler.transform(
        customer_df
    )

    customer_tensor = torch.FloatTensor(
        customer_scaled
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            customer_tensor
        )

        probability = torch.sigmoid(
            logits
        ).item()

    prediction = int(
        probability >= threshold
    )

    return prediction, probability