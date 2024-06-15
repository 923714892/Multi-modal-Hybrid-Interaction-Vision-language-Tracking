from lib.models.language_model.bert import BERT
# from lib.checkpoints.language_model.bert_huggingface import BERT_HUGGINGFACE


def build_bert():
    # position_embedding = build_position_encoding(cfg)
    train_bert = False
    bert_type = 'pytorch'
    if bert_type == "pytorch":
        bert_model = BERT('bert-base-uncased', '/home/data/zgt/CVPR2024/OSTrack-main/pretrained/bert/bert-base-uncased.tar.gz', train_bert,256,
                         36, 12)
    else:
        raise ValueError("Undefined BERT TYPE '%s'" % bert_type)
    return bert_model
