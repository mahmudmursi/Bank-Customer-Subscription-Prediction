from src.predict import predict_customer


def test_prediction():

    assert callable(
        predict_customer
    )