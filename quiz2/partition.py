def partition_by_feature_value(dataset, feature_index):
    partition = []
    keys = []

    for data in dataset:
        features, _ = data
        key = features[feature_index]

        if len(partition) == 0:
            partition.append([data])
            keys.append(key)
        else:
            try:
                i = keys.index(key)
                partition[i].append(data)
            except:
                partition.append([data])
                keys.append(key)

    def separation(x, keyset=keys, i=feature_index):
        return keyset.index(x[i])
    
    return separation, partition 