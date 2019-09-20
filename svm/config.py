tags = ['情绪效价', '情绪用户']

def get_data_path(tag_type):
    origin_file_path = ''
    train_npz_path = ''
    test_npz_path = ''
    if tag_type == '情绪效价':
        origin_file_path = 'data/emotion_titer.xlsx'
        train_npz_path = 'data/emotion_titer_train.npz'
        test_npz_path = 'data/emotion_titer_test.npz'
    elif tag_type == '情绪用户':
        origin_file_path = 'data/emotion_user.xlsx'
        train_npz_path = 'data/emotion_user_train.npz'
        test_npz_path = 'data/emotion_user_test.npz'
    return origin_file_path, train_npz_path, test_npz_path

def get_model_path(tag_type):
    vectorizer_model_path = ''
    reducer_model_path = ''
    classifier_model_path = ''
    if tag_type == '情绪效价':
        vectorizer_model_path = 'models/emotion_titer_vectorizer.pkl.gz'
        reducer_model_path = 'models/emotion_titer_reducer.pkl.gz'
        classifier_model_path = 'models/emotion_titer_classifier.pkl.gz'
    elif tag_type == '情绪用户':
        vectorizer_model_path = 'models/emotion_user_vectorizer.pkl.gz'
        reducer_model_path = 'models/emotion_user_reducer.pkl.gz'
        classifier_model_path = 'models/emotion_user_classifier.pkl.gz'
    return vectorizer_model_path, reducer_model_path, classifier_model_path

def get_category_names(tag_type):
    names = []
    if tag_type == '情绪效价':
        names = ['中性', '负向']
    elif tag_type == '情绪用户':
        names = ['无', '情绪用户']
    return names