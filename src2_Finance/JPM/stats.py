import pandas as pd
import numpy as np

class Period:
    def __init__(self, name) -> None:
        self.name = name
        self.dataset = pd.read_csv(f"{name}/dataset.csv")

    def stats(self) -> None:
        m = np.array(self.dataset["T"])
        m_min, m_max, m_mean = np.min(m), np.max(m), np.mean(m)
        print(f"{self.name}")
        print(f"Total Len: {len(self.dataset)}")
        print(f"moneyness: \nmin:{m_min}\nmax:{m_max}\nmean:{m_mean}")
        print(f"Mode: {np.median(m)}")
        # print(f"Moneyness < 0.5: {(m < 0.5).sum()}")
        # print(f"Moneyness > 1.5: {(m > 1).sum()}")

period1 = Period("Period1")
period1.stats()
