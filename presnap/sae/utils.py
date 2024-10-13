def build_feature_index(vocab, feature_names):
    # Define the feature to embedding index mapping
    feature_index = {}
    for i, feature in enumerate(feature_names):
        if feature not in ['drives']:
            if feature.startswith('home') and f"away{feature[4:]}" in feature_names or feature.startswith('away') and f"home{feature[4:]}" in feature_names:
                index_key = feature[4:]
            else:
                index_key = feature
            feature_index[i] = list(vocab.keys()).index(index_key)
    return feature_index

def build_encoder_config(vocab, feature_names, **kwargs):
    vocab_sizes = [len(values) for key, values in vocab.items() if key != 'drives']
    num_features = len(feature_names)
    feature_index = build_feature_index(vocab, feature_names)
    return vocab_sizes, num_features, feature_index