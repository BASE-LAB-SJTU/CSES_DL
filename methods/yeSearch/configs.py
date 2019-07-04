def get_config():
    conf = {
        'workdir': './data/full_github/',
        'data_params': {
            'train_word_wordembedding': 'word2vec',
            'train_vocabulary': 'vocabulary',
            'train_database': 'train/',
            'train_model': 'word2vec_model_Epoch_',

            'eval_query': 'query2answer.json',
            'origin_codebase_dir': 'codebase/',
            'eval_codebase_dir': 'clean_codebase/',
            'eval_idf': 'idf/',
            'eval_log': 'yesearch_eval.csv',
        },

        'model_params': {
            'iter': 10,
            'epochs': 5,
            'reload': 0
        }
    }
    return conf
