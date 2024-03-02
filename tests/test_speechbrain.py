from speechbrain.inference import EncoderClassifier


def test_import_speechbrain_encoder_classifier(cache):
    """This is a simple test that check if speechbrain
    EncoderClassifier can be imported. It does not check
    if the model is working properly. 
    """

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir=cache,
    )
    assert isinstance(model, EncoderClassifier)
