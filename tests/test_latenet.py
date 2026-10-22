import torch
from torch import nn

from features import CATEGORICAL, NUMERIC
from latenet import LateNet

VOCAB_SIZES = {"route_id": 3, "stop_id": 5, "hour": 24, "day_of_week": 7, "point_type": 3}
DIMS = {"route_id": 8, "stop_id": 16, "hour": 4, "day_of_week": 3, "point_type": 2}


def make(hidden=(64, 32), dropout=0.1):
    torch.manual_seed(0)
    return LateNet(VOCAB_SIZES, DIMS, list(hidden), dropout)


def batch(n):
    g = torch.Generator().manual_seed(1)
    cat = torch.stack([torch.randint(0, VOCAB_SIZES[c] + 1, (n,), generator=g)
                       for c in CATEGORICAL], dim=1)
    num = torch.rand(n, len(NUMERIC), generator=g)
    return cat, num


def test_one_score_per_trip():
    cat, num = batch(6)
    assert make()(cat, num).shape == (6,)


def test_tables_have_one_extra_row_for_unknown():
    model = make()
    for col in CATEGORICAL:
        assert model.embeddings[col].num_embeddings == VOCAB_SIZES[col] + 1


def test_first_layer_takes_embeddings_plus_numeric():
    assert make().mlp[0].in_features == 8 + 16 + 4 + 3 + 2 + 4


def test_hidden_layers_follow_the_config():
    linears = [m for m in make(hidden=(128, 64, 32)).mlp if isinstance(m, nn.Linear)]
    assert [m.out_features for m in linears] == [128, 64, 32, 1]


def test_ends_in_a_linear_layer_not_a_sigmoid():
    assert isinstance(make().mlp[-1], nn.Linear)


def test_unknown_row_is_zero_and_stays_zero_after_a_training_step():
    model = make()
    cat, num = batch(32)
    cat[:4] = 0                                   # four trips where every value is unknown
    loss = nn.BCEWithLogitsLoss()(model(cat, num), torch.ones(32))
    loss.backward()
    torch.optim.Adam(model.parameters(), lr=0.1).step()
    for col in CATEGORICAL:
        assert torch.all(model.embeddings[col].weight[0] == 0)


def test_eval_mode_turns_dropout_off():
    model = make(dropout=0.5)
    cat, num = batch(64)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(cat, num), model(cat, num))
    model.train()
    assert not torch.equal(model(cat, num), model(cat, num))
