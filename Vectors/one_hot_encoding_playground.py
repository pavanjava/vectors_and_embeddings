import numpy as np
from sklearn.preprocessing import OneHotEncoder


class OneHotEncoding:
    def __init__(self):
        self.vocabulary = [["king"], ["queen"], ["man"], ["woman"], ["car"], ["bike"]]
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoded_matrix = self.encoder.fit_transform(self.vocabulary)

    def get_vector(self, word):
        return self.encoded_matrix[self.encoder.categories_[0].tolist().index(word)]

    def cosine_similarity(self, a, b):
        dot_product = np.dot(a, b)
        magnitude_a = np.sqrt(np.sum(a ** 2))
        magnitude_b = np.sqrt(np.sum(b ** 2))
        return dot_product / (magnitude_a * magnitude_b)


if __name__ == "__main__":

    ohe = OneHotEncoding()

    print("=" * 55)
    print("  VOCABULARY")
    print("=" * 55)
    print(f"  Words : {[w[0] for w in ohe.vocabulary]}")

    print("\n" + "=" * 55)
    print("  ONE-HOT ENCODED VECTORS")
    print("=" * 55)
    for word in [w[0] for w in ohe.vocabulary]:
        vec = ohe.get_vector(word)
        print(f"  {word:<8} : {vec.astype(int)}")

    print("\n" + "=" * 55)
    print("  COSINE SIMILARITY (one-hot vectors)")
    print("=" * 55)
    king_vec  = ohe.get_vector("king")
    queen_vec = ohe.get_vector("queen")
    car_vec   = ohe.get_vector("car")

    sim_king_queen = ohe.cosine_similarity(king_vec, queen_vec)
    sim_king_car   = ohe.cosine_similarity(king_vec, car_vec)

    print(f"  King  vs Queen : {sim_king_queen:.4f}")
    print(f"  King  vs Car   : {sim_king_car:.4f}")

    print("\n  Observation:")
    print("  King vs Queen and King vs Car are IDENTICAL scores!")
    print("  One-hot has no concept of semantic closeness.")
    print("  Every word is equally distant from every other word.")
    print("=" * 55)

    print("\n" + "=" * 55)
    print("  WHY ONE-HOT FAILS")
    print("=" * 55)
    print(f"  Vector size = vocabulary size : {len(ohe.vocabulary)} dimensions")
    print(f"  Non-zero values per vector    : 1 out of {len(ohe.vocabulary)}")
    print(f"  Sparsity                      : {((len(ohe.vocabulary)-1)/len(ohe.vocabulary))*100:.1f}% zeros")
    print("  Semantic relationship         : None")
    print("=" * 55)