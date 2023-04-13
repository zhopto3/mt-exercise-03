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
    ValData = {"Valid. Perplexity":["Epoch " + str(num) for num in range(1,41)]}
    TrainData = {"Training Perplexity":["Epoch " + str(num) for num in range(1,41)]}
    TestData = {"Test Perplexity":["Epoch " + str(num) for num in range(1,41)]}
    #Append the end-of-training test data to the test set perplexity table
    TestData["Test Perplexity"].append("End of Training") # type: ignore
    for file in files:
        input = pd.read_csv(file)
        dropout = file.rstrip("perplexity.csv")[-1]
        ValData["Dropout 0."+str(dropout)]=list(input["Valid. Perplexity"])[:40]
        TrainData["Dropout 0."+str(dropout)]=list(input["Training Perplexity"])[:40]
        TestData["Dropout 0."+str(dropout)]=list(input["Test Perplexity"])[:40]
        TestData["Dropout 0."+str(dropout)].append(list(input["Valid. Perplexity"])[-1])

    TestTable = pd.DataFrame(TestData).to_csv("Testppy_table.csv", index=False)
    ValTable = pd.DataFrame(ValData).to_csv("Valppy_table.csv", index=False)
    TrainTable = pd.DataFrame(TrainData).to_csv("Trainppy_table.csv", index=False)


def get_chart(file:str)->None:
    data = {"Epoch":[num for num in range(1,41)]}
    #Test perplexity not displayed
    input = pd.read_csv(file)
    dropout = file.rstrip("perplexity.csv")[-1]
    data["Validation Perplexity"]=list(input["Valid. Perplexity"])[:40]
    data["Training Perplexity"]=list(input["Training Perplexity"])[:40]
    df = pd.DataFrame(data)
    plt.plot(df["Epoch"], df["Validation Perplexity"], label = "Validation")
    plt.plot(df["Epoch"], df["Training Perplexity"], label = "Training")
    plt.ylabel("Perplexity")
    plt.xlabel("Epoch")
    plt.title("Validation and Training Data Perplexity with Dropout of 0."+str(dropout))
    plt.tight_layout()
    plt.legend()
    plt.savefig("Dropout"+str(dropout)+"Fig.png")
    plt.close()


def main():
    args = get_args()

    #make table
    get_table(args.perplexity_logs)
    #make graphs
    for files in args.perplexity_logs:
        get_chart(files)


if __name__ == "__main__":
    main()
