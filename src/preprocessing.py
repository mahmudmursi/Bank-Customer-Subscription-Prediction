def clean_pdays(df):

    df["pdays_clean"] = (
        df["pdays"]
        .replace(999, None)
    )

    return df