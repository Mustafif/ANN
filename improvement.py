import pandas as pd

def improvement(folder):
    df = pd.read_csv(f"{folder}/report.csv")
    df.to_latex(f"{folder}/report.tex", index=False)
    output_path = f"{folder}/improve.csv"
    ml = df["Mean Loss"]
    in_sample = 1 - (ml[2] / ml[0])
    out_sample = 1 - (ml[3] / ml[1])

    result = {
        "": ["w. D-Layer", "w/o D-Layer", "Improvement Ratio"],
        "In-Sample": [f"{ml[2]:.6e}", f"{ml[0]:.6e}", f"{in_sample.round(4)}"],
        "Out-of-Sample": [f"{ml[3]:.6e}", f"{ml[1]:.6e}", f"{out_sample.round(4)}"]
    }


    df2 = pd.DataFrame(result)
    df2.to_csv(output_path, index=False)
    df2.to_latex(f"{folder}/improve.tex", index=False)

folders = [
    "report_HN2_20K", "report_HN2_50K", "report_HN2_100K", "report_Duan2_20K", "report_Duan2_50K", "report_Duan2_100K"
]

for f in folders:
    improvement(f)
