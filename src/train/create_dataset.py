from itertools import permutations
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict


def filter_df(df):
    df = df.fillna("")

    # if "subject" in df.columns:
    #     df = df[["original_text", "rewrite_prompt", "rewritten_text", "subject"]].reset_index(
    #         drop=True
    #     )

    # else:
    #     df = df[["original_text", "rewrite_prompt", "rewritten_text"]].reset_index(drop=True)


    df = df[["original_text", "rewrite_prompt", "rewritten_text", "subject"]].reset_index(drop=True)
    df["original_text"] = df["original_text"].apply(lambda x: str(x).strip())
    df["rewritten_text"] = df["rewritten_text"].apply(lambda x: str(x).strip())
    df["rewrite_prompt"] = df["rewrite_prompt"].apply(lambda x: str(x).strip())
    df["subject"] = df["subject"].apply(lambda x: str(x).strip())

    df = df[df["original_text"].apply(lambda x: len(x) >= 200 and len(x) <= 3000)].reset_index(
        drop=True
    )
    df = df[df["rewritten_text"].apply(lambda x: len(x) >= 200 and len(x) <= 3000)].reset_index(
        drop=True
    )
    df = df[df["rewrite_prompt"].apply(lambda x: len(x) >= 5 and len(x) <= 500)].reset_index(
        drop=True
    )
    # df = df[df["is_well_written"].apply(lambda x: bool(x))].reset_index(drop=True)

    df = df[df["subject"].apply(lambda x: len(x) >= 3 and len(x) <= 200)].reset_index(
        drop=True
    )

    return df


def get_df_train():
    data_list = [
        "/kaggle/input/gemma_rewritten_text_exllama/proc_dataset_updated.csv",
        "/kaggle/input/pedro-data/data_subject.csv",
        "/kaggle/input/pedro-data/data_subject_2.csv",
        "/kaggle/input/pedro-data/data_subject_3.csv",
    ]
    
    # data_list = [
    #     "/kaggle/input/gemma_rewritten_text_exllama_bkp/proc_dataset_updated.csv"
    # ]
    
    # df = pd.read_csv("/kaggle/input/gemma_rewritten_text_exllama/proc_dataset_updated.csv")
    # df = pd.read_csv("/kaggle/input/gemma_rewritten_text_exllama/proc_dataset_updated_evaluated.csv")

    df = pd.concat([pd.read_csv(data) for data in data_list], ignore_index=True)
    # df = pd.read_csv(data_list[0])
    df = filter_df(df)

    return df


def get_df_val():
    df = pd.read_parquet("/kaggle/input/df_with_emb.parquet")
    df = filter_df(df)
    df = df.sample(500, random_state=42)
    return df


def get_perm(df):
    new_rows = []

    for idx, row in df.iterrows():
        subject = row["subject"]
        subject_list = [x.strip() for x in subject.split(",")]

        for perm in list(permutations(subject_list))[:3]:
            new_row = row.to_dict()
            new_row["subject"] = ", ".join(perm)
            new_rows.append(new_row)

    df = pd.DataFrame(new_rows)
    return df


df_train = get_df_train()
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

val_sz = 500
df_val = df_train.iloc[-val_sz:].copy().reset_index(drop=True)
df_train = df_train.iloc[:-val_sz].copy().reset_index(drop=True)
# df_val = get_df_val()


# df_train = get_perm(df_train)
# df_val = get_perm(df_val)


train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)

dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})


dataset_dict.save_to_disk("/kaggle/input/llm-prompt-rewriting-dataset")
