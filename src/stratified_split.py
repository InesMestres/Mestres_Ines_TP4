import numpy as np

def stratified_split(labels, test_size=0.2):
    generator = np.random.default_rng(60)
    train_indxs = []
    test_indxs = []
    for class_label in np.unique(labels):
        class_indxs = np.where(labels == class_label)[0]
        generator.shuffle(class_indxs)
        test_count = int(len(class_indxs)*test_size)
        test_indxs.extend(class_indxs[:test_count])
        train_indxs.extend(class_indxs[test_count:])
    train_indxs = np.array(train_indxs)
    test_indxs = np.array(test_indxs)
    generator.shuffle(train_indxs)
    generator.shuffle(test_indxs)
    return train_indxs, test_indxs
