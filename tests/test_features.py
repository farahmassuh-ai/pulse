import pandas as pd
import pytest

from features import (CATEGORICAL, NUMERIC, UNKNOWN, build_vocab, check_inputs, encode,
                      fit_scaler, numeric_inputs)


def rows(stops, hours):
    n = len(stops)
    return pd.DataFrame({
        "route_id": ["01"] * n,
        "stop_id": stops,
        "hour": pd.array(hours, dtype="int32"),
        "day_of_week": pd.array([0] * n, dtype="int32"),
        "point_type": ["Midpoint"] * n,
    })


def test_vocab_starts_at_one_and_keeps_zero_free():
    vocab = build_vocab(rows(["a", "b", "a"], [7, 8, 7]))
    assert vocab["stop_id"] == {"a": 1, "b": 2}
    assert all(UNKNOWN not in v.values() for v in vocab.values())


def test_vocab_comes_from_train_only():
    train, val = rows(["a"], [7]), rows(["z"], [9])
    vocab = build_vocab(train)
    assert "z" not in vocab["stop_id"] and "9" not in vocab["hour"]
    assert encode(val, vocab)["stop_id"].tolist() == [UNKNOWN]


def test_encode_gives_plain_integers_in_every_column():
    train = rows(["a", "b"], [7, 8])
    encoded = encode(train, build_vocab(train))
    assert list(encoded.columns) == CATEGORICAL
    assert all(encoded[c].dtype == "int64" for c in CATEGORICAL)
    assert encoded["stop_id"].tolist() == [1, 2]


def test_hour_matches_whatever_integer_type_it_arrives_as():
    vocab = build_vocab(rows(["a"], [7]))
    serving = rows(["a"], [7]).astype({"hour": "int64"})
    assert encode(serving, vocab)["hour"].tolist() == [1]


# ---------- guards ----------

def clean_rows():
    df = rows(["a", "b"], [7, 8])
    df["time_point_order"] = pd.array([1, 2], dtype="Int64")
    df["direction_id"] = ["Inbound", "Outbound"]
    df["standard_type"] = ["Headway", "Schedule"]
    return df


def test_clean_rows_pass():
    check_inputs(clean_rows())          # no error raised


def test_float_index_column_is_caught():
    df = clean_rows().astype({"hour": "float64"})
    with pytest.raises(ValueError, match="hour is a float column"):
        check_inputs(df)


def test_missing_time_point_order_is_caught():
    df = clean_rows()
    df.loc[0, "time_point_order"] = pd.NA
    with pytest.raises(ValueError, match="time_point_order has 1 missing"):
        check_inputs(df)


def test_unexpected_flag_value_is_caught():
    df = clean_rows()
    df.loc[0, "standard_type"] = "Frequency"
    with pytest.raises(ValueError, match="standard_type has unexpected values"):
        check_inputs(df)


def test_missing_flag_value_is_caught():
    df = clean_rows()
    df.loc[0, "direction_id"] = None
    with pytest.raises(ValueError, match="direction_id has 1 missing"):
        check_inputs(df)


# ---------- flags and scaling ----------

def test_flags_follow_the_written_mapping():
    df = clean_rows()                                   # Inbound/Headway, Outbound/Schedule
    df["day_of_week"] = pd.array([4, 5], dtype="int32")  # Friday, Saturday
    out = numeric_inputs(df, {"min": 1.0, "max": 2.0})
    assert out["direction"].tolist() == [1, 0]
    assert out["standard_type"].tolist() == [0, 1]
    assert out["weekend"].tolist() == [0, 1]


def test_scaler_comes_from_train_and_is_plain_floats():
    train = clean_rows()
    train["time_point_order"] = pd.array([1, 14], dtype="Int64")
    scaler = fit_scaler(train)
    assert scaler == {"min": 1.0, "max": 14.0}
    assert all(type(v) is float for v in scaler.values())


def test_min_max_scaling_maps_train_range_to_zero_one():
    df = clean_rows()
    df["time_point_order"] = pd.array([1, 14], dtype="Int64")
    out = numeric_inputs(df, {"min": 1.0, "max": 14.0})
    assert out["time_point_order"].tolist() == [0.0, 1.0]


def test_value_beyond_train_range_goes_past_one():
    df = clean_rows()
    df["time_point_order"] = pd.array([1, 15], dtype="Int64")
    out = numeric_inputs(df, {"min": 1.0, "max": 14.0})
    assert out["time_point_order"].iloc[1] > 1.0


def test_numeric_inputs_are_float32_in_fixed_order():
    out = numeric_inputs(clean_rows(), {"min": 1.0, "max": 2.0})
    assert list(out.columns) == NUMERIC
    assert all(out[c].dtype == "float32" for c in NUMERIC)


def test_constant_time_point_order_cannot_be_scaled():
    train = clean_rows()
    train["time_point_order"] = pd.array([3, 3], dtype="Int64")
    with pytest.raises(ValueError, match="constant"):
        fit_scaler(train)


def test_vocab_and_encode_can_leave_a_column_out():
    train = rows(["a", "b"], [7, 8])
    vocab = build_vocab(train, ["route_id", "stop_id", "hour", "day_of_week"])
    assert list(vocab) == ["route_id", "stop_id", "hour", "day_of_week"]
    assert list(encode(train, vocab).columns) == ["route_id", "stop_id", "hour", "day_of_week"]
