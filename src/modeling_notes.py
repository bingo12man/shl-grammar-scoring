"""Reference configuration for the strongest submitted architecture."""

V11_WEIGHTS = {
    "wavlm_layer9": 0.50,
    "hubert": 0.20,
    "grammar_embedding": 0.20,
    "handcrafted_audio": 0.10,
}

V13_WEIGHTS = {
    "v11_multimodal": 0.70,
    "v12_cross_attention": 0.30,
}


def blend_v13(v11_prediction, v12_prediction):
    return (
        V13_WEIGHTS["v11_multimodal"] * v11_prediction
        + V13_WEIGHTS["v12_cross_attention"] * v12_prediction
    )
