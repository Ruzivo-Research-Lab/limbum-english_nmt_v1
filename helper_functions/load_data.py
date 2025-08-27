from huggingface_hub import hf_hub_download
from huggingface_hub import login
# Key
login("YOUR_HUGGINGFACE_KEY")


en_lmb_dic = hf_hub_download(repo_id="ruzivo/limbum_english_sentences", repo_type="dataset", filename="english_limbum_dictionary.tsv")
en_lmb_nt = hf_hub_download(repo_id="ruzivo/english_limbum_new_testament", repo_type="dataset", filename="english_limbum_new_testament.tsv")

# Loading each file separately to be sure I am not tripping
df_dic = pd.read_csv(en_lmb_dic, sep="\t", dtype=str)
df_nt = pd.read_csv(en_lmb_nt, sep="\t", dtype=str)

print("Dictionary file shape:", df_dic.shape)
print("New Testament file shape:", df_nt.shape)


def load_and_clean(path, lim_col, eng_col):
    df = pd.read_csv(path, sep="\t", dtype=str)
    # drop auto "Unnamed: *" columns if present
    df = df.drop(columns=[c for c in df.columns if c.lower(
    ).startswith("unnamed")], errors="ignore")
    # rename to standard short names
    df = df.rename(columns={lim_col: "limbum", eng_col: "english"})
    # normalize + strip
    df["limbum"] = df["limbum"].apply(
        lambda s: normalize("NFC", str(s).strip()))
    df["english"] = df["english"].astype(str).str.strip()
    # keep rows where both sides are present (no extra filtering)
    df = df.dropna(subset=["limbum", "english"])
    return df[["limbum", "english"]]


# Load both datasets with the explicit column names
df_dic = load_and_clean(en_lmb_dic, "Limbum", "English")
df_nt = load_and_clean(en_lmb_nt,  "verse_text_limbum", "verse_text_english")

print("Dictionary pairs (post-clean):", df_dic.shape)
print("New Testament pairs (post-clean):", df_nt.shape)

# Concatenate, remove exact duplicates, and shuffle
df_all = pd.concat([df_dic, df_nt], ignore_index=True)
before_dups = len(df_all)
df_all = df_all.drop_duplicates().reset_index(drop=True)
after_dups = len(df_all)
df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)

print("Merged df_all size before dropping duplicates:", before_dups)
print("Merged df_all size after dropping duplicates :", after_dups)

# train and test slices (Limbum \t English)
n_train = int(0.9 * len(df_all))
train = df_all.iloc[:n_train]
test = df_all.iloc[n_train:]

# IMPORTANT: write as tab-separated with NO headers
train.to_csv("limbum_english_training_dataset.txt",
             sep="\t", header=False, index=False)
test.to_csv("limbum_english_test_dataset.txt",
            sep="\t", header=False, index=False)

print("\nWrote files:")
print(" - limbum_english_training_dataset.txt :", len(train))
print(" - limbum_english_test_dataset.txt     :", len(test))


