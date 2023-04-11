import pandas as pd
import matplotlib.pyplot as plt
import argparse
"""This script creates a table that compares models' validation perplexities at each epoch during training. It
also visualizes the change in perplexity over the epochs by outputing line charts for each model.

Authors: Zachary W. Hopton (22-737-027), Allison Ponce de Leon Diaz (22-740-633)

Example call: python3 scripts/visualize_ppy.py --perplexity-logs "./models/logs/point0perplexity.csv" "./models/logs/point2perplexity.csv" "./models/logs/point4perplexity.csv" "./models/logs/point6perplexity.csv" "./models/logs/point8perplexity.csv"
"""

def get_args():
    parser = argparse.ArgumentParser("A script to visualize the perplexity throughout model training")
    parser.add_argument("--perplexity-logs", type = str, nargs = "+",
                        help = "Give the names of any perplexity logs you're interested in visualizing")
    
    return parser.parse_args()


def get_table(files: list)->None:
    data = {"Valid. Perplexity":["Epoch" + str(num) for num in range(1,41)]}
    data["Valid. Perplexity"].append("End of Training") # type: ignore
    for file in files:
        input = pd.read_csv(file)
        dropout = file.rstrip("perplexity.csv")[-1]
        data["Dropout 0."+str(dropout)]=list(input["Valid. Perplexity"])
    table = pd.DataFrame(data).to_csv("ppy_table.csv", index=False)


def get_chart(file:str)->None:
    data = {"Epoch":[num for num in range(1,42)]}
    #data["Epoch"].append("End of Training") # type: ignore 
    #The "41st" epoch is the test set...
    input = pd.read_csv(file)
    dropout = file.rstrip("perplexity.csv")[-1]
    data["Perplexity"]=list(input["Valid. Perplexity"])
    df = pd.DataFrame(data)
    plt.plot(df["Epoch"], df["Perplexity"])
    plt.ylabel("Perplexity")
    plt.xlabel("Epoch")
    plt.title("Validation Data Perplexity for Dropout of 0."+str(dropout))
    plt.tight_layout()
    #STILL NEED TO ADD THE SECOND LINE FOR TRAINING PERPLEXITY
    plt.savefig("Dropout"+str(dropout)+"Fig.png")
    plt.close()


def main():
    args = get_args()
    #print(args.perplexity_logs)
    #make table
    get_table(args.perplexity_logs)
    #make graphs
    for files in args.perplexity_logs:
        get_chart(files)


if __name__ == "__main__":
    main()
