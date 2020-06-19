%%time
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
                        
rows = zip(df_train['comment_text'].values.tolist(), df_train.toxic.values.tolist())
train = Parallel(n_jobs=4, backend='multiprocessing')(delayed(regular_encode)(row) for row in tqdm(rows))

rows = zip(df_valid['comment_text'].values.tolist(), df_valid.toxic.values.tolist())
valid = Parallel(n_jobs=4, backend='multiprocessing')(delayed(regular_encode)(row) for row in tqdm(rows))
                        
rows = test.content.values.tolist()
x_test = Parallel(n_jobs=4, backend='multiprocessing')(delayed(regular_encode)(row,is_test=True) for row in tqdm(rows))

x_train = np.vstack(np.array(train)[:,0])
y_train = np.array(train)[:,1].astype(np.int32)
x_valid = np.vstack(np.array(valid)[:,0])
y_valid = np.array(valid)[:,1].astype(np.int32)
x_train.shape,y_train.shape,x_valid.shape,y_valid.shape
