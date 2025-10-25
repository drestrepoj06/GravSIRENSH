"""Generate 250k uniformly distributed (Fibonacci) random samples on the
sphere that contain lon, lat, r, potential and acceleration from EGM2008
jhonr"""

import os
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Generate_data import GravityDataGenerator


def main():
    generator = GravityDataGenerator(
        lmax_full=2190,
        lmax_base=2,
        n_samples=250_000,
        mode="test",
        output_dir=os.path.join(os.path.dirname(BASE_DIR), "Data"),
        altitude=0.0
    )

    df_test = generator.generate()
    print(df_test.head())
    print(f"✅ Test data shape: {df_test.shape}")


if __name__ == "__main__":
    main()