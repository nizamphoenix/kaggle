from joblib import Parallel, delayed
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('jplu/tf-xlm-roberta-base')

def regular_encode(row, is_test=False, maxlen=MAX_LEN):
    outp = None
    if is_test:#for test data
        enc_di = tokenizer.encode_plus(
            str(row),#row:text
            return_attention_masks=False, 
            return_token_type_ids=False,
            pad_to_max_length=True,
            max_length=maxlen
        )
        outp = np.array(enc_di['input_ids'])
    else:#for validation/train data
        enc_di = tokenizer.encode_plus(
            str(row[0]),#row:(text,label)
            return_attention_masks=False, 
            return_token_type_ids=False,
            pad_to_max_length=True,
            max_length=maxlen
        )
        outp = np.array(enc_di['input_ids']), row[1]
    return outp
