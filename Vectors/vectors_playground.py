import numpy as np

print("Welcome to the world of vectors")

class Vectors:
    def __init__(self):
        # ── 1. What is a vector? ──────────────────────────────────────────────────
        self.king   = np.array([1.0, 0.0, 1.0])   # manually assigned: [royalty, gender, power]
        self.queen  = np.array([1.0, 1.0, 1.0])   # [royalty, gender, power]
        self.car    = np.array([0.0, 0.0, 0.3])   # [royalty, gender, power] -- unrelated word

    # ── 2. Magnitude (length of a vector) ────────────────────────────────────
    def compute_magnitude(self, v):
        return np.sqrt(np.sum(v ** 2))

    # ── 3. Cosine Similarity (from scratch) ───────────────────────────────────
    def cosine_similarity(self, a, b):
        dot_product  = np.dot(a, b)
        similarity   = dot_product / (self.compute_magnitude(a) * self.compute_magnitude(b))
        return similarity


if __name__ == "__main__":
    vectors = Vectors()

    king = vectors.king
    queen = vectors.queen
    car = vectors.car

    print("=" * 50)
    print("  RAW VECTORS")
    print("=" * 50)
    print(f"  King  : {king}")
    print(f"  Queen : {queen}")
    print(f"  Car   : {car}")


    print("\n" + "=" * 50)
    print("  MAGNITUDE (length of each vector)")
    print("=" * 50)
    print(f"  |King|  : {vectors.compute_magnitude(king):.4f}")
    print(f"  |Queen| : {vectors.compute_magnitude(queen):.4f}")
    print(f"  |Car|   : {vectors.compute_magnitude(car):.4f}")



    print("\n" + "=" * 50)
    print("  COSINE SIMILARITY (from scratch)")
    print("=" * 50)
    sim_king_queen = vectors.cosine_similarity(king, queen)
    sim_king_car   = vectors.cosine_similarity(king, car)
    sim_queen_car  = vectors.cosine_similarity(queen, car)

    print(f"  King  vs Queen : {sim_king_queen:.4f}  <-- similar")
    print(f"  King  vs Car   : {sim_king_car:.4f}  <-- dissimilar")
    print(f"  Queen vs Car   : {sim_queen_car:.4f}  <-- dissimilar")

    print("\n  Observation:")
    print("  King and Queen are closer to each other")
    print("  than either is to Car -- geometry encodes meaning!")
    print("=" * 50)
