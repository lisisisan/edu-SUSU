from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer

def build_model(encoder_name="google/vit-base-patch16-224-in21k",
                decoder_name="gpt2"):
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_name, decoder_name)
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)

    # GPT2 не имеет pad_token — добавляем
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.decoder.resize_token_embeddings(len(tokenizer))

    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer
