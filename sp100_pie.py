import matplotlib.pyplot as plt
import pandas as pd

# Read your CSV
df = pd.read_csv("/home/mustafif/Downloads/archive/SP500.csv")

# Count companies by sector
sector_counts = df["GICS Sector"].value_counts()

# Plot
plt.figure(figsize=(8, 8))
sector_counts.plot(kind="pie", autopct="%1.1f%%", startangle=90)

plt.ylabel("")
plt.title("Distribution of S&P 100 Companies by Sector")
plt.tight_layout()
plt.show()
