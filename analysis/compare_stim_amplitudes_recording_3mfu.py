"""Compares the stimulation amplitudes for the recordings and 3MFU."""

import numpy as np
import pandas as pd
from scipy.stats import sem


# Recording and 3MFU stimulation amplitudes
amplitudes = {
    "sub": [
        "EL006",
        "EL007",
        "EL009",
        "EL012",
        "EL014",
        "EL017",
        "EL019",
        "EL022",
        "EL023",
        "EL025",
        "EL026",
        "EL027",
    ],
    "recording": [
        [np.nan, 2.0],  # 006
        [2.0, 1.5],  # 007
        [2.0, 2.0],  # 009
        [1.5, 1.5],  # 012
        [2.0, 2.5],  # 014
        [3.0, 2.0],  # 017
        [np.nan, 2.5],  # 019
        [3.0, 3.0],  # 022
        [2.5, 2.5],  # 023
        [3.0, 3.0],  # 025
        [3.0, 2.5],  # 026
        [3.0, 3.0],  # 027
    ],
    "3MFU": [
        [np.nan, np.nan],  # 006
        [np.nan, np.nan],  # 007
        [np.nan, np.nan],  # 009
        [1.0, 1.7],  # 012
        [1.5, np.nan],  # 014
        [2.1, 2.3],  # 017
        [1.7, 2.0],  # 019
        [1.6, 2.0],  # 022
        [1.9, 1.7],  # 023
        [2.8, 2.8],  # 025
        [np.nan, np.nan],  # 026
        [1.9, 2.0],  # 027
    ],
}
amplitudes = pd.DataFrame(amplitudes)


# Compare values
mean_rec = np.nanmean(amplitudes["recording"].tolist(), axis=1)
mean_mfu = np.nanmean(amplitudes["3MFU"].tolist(), axis=1)

print(
    f"Recording mean amplitude: {np.nanmean(mean_rec):.1f} +/- "
    f"{sem(mean_rec, nan_policy='omit'):.1f} (mA)"
)
print(
    f"3MFU mean amplitude: {np.nanmean(mean_mfu):.1f} +/- "
    f"{sem(mean_mfu, nan_policy='omit'):.1f} (mA)"
)
