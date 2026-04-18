# Import packages 
# # Loading the files and exploring the data with pandas and sklearn
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import umap.umap_ as umap


# %%
# Load the data
####################################################
data = pd.read_csv(
    #'C:\\Users\\Jmarc\\Desktop\\Comp BME\\Module-4-Cancer\\data\\TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)  # can also use larger dataset with more genes
    'C:\\Users\\yqr8pz\\Documents\\BME 2315\\Module-4-Cancer\\data\\TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)
metadata_df = pd.read_csv(
    #'C:\\Users\\Jmarc\\Desktop\\Comp BME\\Module-4-Cancer\\data\\TRAINING_SET_GSE62944_metadata.csv', index_col=0, header=0)
    'C:\\Users\\yqr8pz\\Documents\\BME 2315\\Module-4-Cancer\\data\\TRAINING_SET_GSE62944_metadata.csv', index_col=0, header=0)
print(data.head())

# Original Code
# Exploratory data analysis (EDA) on a cancer dataset

# %%
# Explore the data
####################################################
print(data.shape)
print(data.info())
print(data.describe())

# %%
# Explore the metadata
####################################################
print(metadata_df.info())
print(metadata_df.describe())

# %%
# Subset the data for a specific cancer type
####################################################
cancer_type = 'LUAD'  # Lung Adenocarcinoma

# From metadata, get the rows where "cancer_type" is equal to the specified cancer type
# Then grab the index of this subset (these are the sample IDs)
cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
print(cancer_samples)
# Subset the main data to include only these samples
# When you want a subset of columns, you can pass a list of column names to the data frame in []
LUAD_data = data[cancer_samples]

# %%
# Subset by index (genes)
####################################################
desired_gene_list = ['EGFR', 'JAK1', 'JAK2', 'MTOR', 'PIK3CA', 'PIK3CB']
gene_list = [gene for gene in desired_gene_list if gene in LUAD_data.index]
for gene in desired_gene_list:
    if gene not in gene_list:
        print(f"Warning: {gene} not found in the dataset.")

# .loc[] is the method to subset by index labels
# .iloc[] will subset by index position (integer location) instead
LUAD_gene_data = LUAD_data.loc[gene_list]
print(LUAD_gene_data.head())

# Check number of data points per gene
gene_summary = pd.DataFrame({
    "non_missing_count": LUAD_gene_data.count(axis=1),
    "missing_count": LUAD_gene_data.isna().sum(axis=1)
})
print("HERE IS THE GENE SUMMARY:")
print(gene_summary)
# %%
# Basic statistics on the subsetted data
####################################################
print(LUAD_gene_data.describe())
print(LUAD_gene_data.var(axis=1))  # Variance of each gene across samples
# Mean expression of each gene across samples
print(LUAD_gene_data.mean(axis=1))
# Median expression of each gene across samples
print(LUAD_gene_data.median(axis=1))

# %%
# Explore categorical variables in metadata
####################################################
# groupby allows you to group on a specific column in the dataset,
# and then print out summary stats or counts for other columns within those groups
print(metadata_df.groupby('cancer_type')["ajcc_pathologic_tumor_stage"].value_counts())

# Explore average age at diagnosis by cancer type
metadata_df['age_at_diagnosis'] = pd.to_numeric(
    metadata_df['age_at_diagnosis'], errors='coerce')
print(metadata_df.groupby(
    'cancer_type')["age_at_diagnosis"].mean())
# %%
# Merging datasets
####################################################
# Merge the subsetted expression data with metadata for LUAD samples,
# so rows are samples and columns include gene expression for EGFR and MYC and metadata
LUAD_metadata = metadata_df.loc[cancer_samples]
LUAD_merged = LUAD_gene_data.T.merge(
    LUAD_metadata, left_index=True, right_index=True)
print(LUAD_merged.head())

# %%
# Plotting
####################################################
# Boxplot of EGFR expression in LUAD samples using SEABORN
# Works really well with pandas dataframes, because most methods allow you to pass in a dataframe directly
sns.boxplot(data=LUAD_merged, x="ajcc_pathologic_tumor_stage", y='EGFR')
plt.title("EGFR Expression by Tumor Stage in LUAD Samples")
plt.show()

sns.boxplot(data=LUAD_merged, x="ajcc_pathologic_tumor_stage", y='JAK1')
plt.title("JAK1 Expression by Tumor Stage in LUAD Samples")
plt.show()

sns.boxplot(data=LUAD_merged, x="ajcc_pathologic_tumor_stage", y='JAK2')
plt.title("JAK2 Expression by Tumor Stage in LUAD Samples")
plt.show()

sns.boxplot(data=LUAD_merged, x="ajcc_pathologic_tumor_stage", y='MTOR')
plt.title("MTOR Expression by Tumor Stage in LUAD Samples")
plt.show()

sns.boxplot(data=LUAD_merged, x="ajcc_pathologic_tumor_stage", y='PIK3CA')
plt.title("PIK3CA Expression by Tumor Stage in LUAD Samples")
plt.show()

sns.boxplot(data=LUAD_merged, x="ajcc_pathologic_tumor_stage", y='PIK3CB')
plt.title("PIK3CB Expression by Tumor Stage in LUAD Samples")
plt.show()

# Boxplot of MYC and EGFR expression in LUAD samples using PANDAS directly
LUAD_merged[['EGFR', 'JAK1', 'JAK2']].plot.box()
plt.title("EGFR, JAK1, and JAK2 Expression in LUAD Samples")
plt.show()

#UMAP considering both hallmarks
# %%
# Load gene list from hallmarks file
####################################################
gene_list = pd.read_csv(
    r'C:\Users\Jmarc\Desktop\Comp BME\Module-4-Cancer\Menyhart_JPA_CancerHallmarks_core.txt',
    sep='\t', header=None, index_col=0)
print(gene_list)

# Pull the immune evasion and angiogenesis gene sets from the hallmarks file by their category name
immune_list = list(gene_list.loc['EVADING IMMUNE DESTRUCTION'])
angio_list  = list(gene_list.loc['INDUCING ANGIOGENESIS'])

# Some rows have empty cells due to uneven gene counts — remove those NaN values
immune_list = [g for g in immune_list if pd.notna(g)]
angio_list  = [g for g in angio_list  if pd.notna(g)]

# Combine both gene lists into one master list
all_genes = immune_list + angio_list
print(f"Immune genes: {immune_list}")
print(f"Angiogenesis genes: {angio_list}")

# %%
# Subset to LUAD + filter to genes present in dataset
# Grab only the LUAD (Lung Adenocarcinoma) samples from the full dataset
cancer_samples = metadata_df[metadata_df['cancer_type'] == 'LUAD'].index
LUAD_data      = data[cancer_samples]
LUAD_metadata  = metadata_df.loc[cancer_samples].copy()

# Keep only the ones that will be in the expression dataset
gene_list_found = [g for g in all_genes if g in LUAD_data.index]
missing_genes   = [g for g in all_genes if g not in LUAD_data.index]
print(f"\nGenes found in dataset: {len(gene_list_found)}")
print(f"Genes missing from dataset: {missing_genes}")

# Tumor stages come in sub-stages like "Stage IA" — simplify them to just Stage I, II, III, IV
def simplify_stage(stage):
    if pd.isna(stage): return None
    s = str(stage)
    if 'IV'  in s: return 'Stage IV'
    if 'III' in s: return 'Stage III'
    if 'II'  in s: return 'Stage II'
    if 'I'   in s: return 'Stage I'
    return None

LUAD_metadata['simple_stage'] = LUAD_metadata['ajcc_pathologic_tumor_stage'].apply(simplify_stage)
# Drop any samples where the stage is unknown
LUAD_metadata_clean = LUAD_metadata.dropna(subset=['simple_stage'])
clean_samples = LUAD_metadata_clean.index

# Subset the expression matrix to our genes and cleaned samples, then flip so samples are rows
LUAD_gene_data = LUAD_data.loc[gene_list_found, clean_samples]
LUAD_merged    = LUAD_gene_data.T.merge(
    LUAD_metadata_clean[['simple_stage']], left_index=True, right_index=True)

# Fill in any missing expression values with the average for that gene, then standardize
# Standardizing puts all genes on the same scale so no single gene dominates the analysis
X = LUAD_merged[gene_list_found].values
X = SimpleImputer(strategy='mean').fit_transform(X)
X_scaled = StandardScaler().fit_transform(X)

stage_labels = LUAD_merged['simple_stage'].values
stage_order  = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
# Assign a color to each stage for plotting
palette      = {'Stage I': '#4CAF50', 'Stage II': '#2196F3',
                'Stage III': '#FF9800', 'Stage IV': '#F44336'}

# %%
# UMAP
####################################################
# UMAP reduces our many genes down to 2 dimensions so we can plot and visually inspect
# whether samples with similar gene expression naturally group together by tumor stage
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap  = reducer.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
# Each dot is one patient sample, colored by their tumor stage
plt.scatter(X_umap[:, 0], X_umap[:, 1],
            c=[list(palette.values())[stage_order.index(s)] for s in stage_labels],
            s=60, alpha=0.8)

# Build a legend manually since we're using plt.scatter instead of seaborn
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=palette[s], label=s) for s in stage_order]
plt.legend(handles=legend_elements, title='Tumor Stage',
           bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("UMAP Projection of LUAD Samples\n(Immune Evasion + Angiogenesis Genes)")
plt.tight_layout()
plt.savefig('umap_luad.png', dpi=150, bbox_inches='tight')
plt.show()


# UMAP Seperating Hallmarks
# %%
# UMAP — Immune Evasion Genes Only
# Filter down to only the immune evasion genes that were found in the dataset
immune_found = [g for g in immune_list if g in gene_list_found]
X_immune = LUAD_merged[immune_found].values
# Fill missing values and standardize just like before
X_immune = SimpleImputer(strategy='mean').fit_transform(X_immune)
X_immune_scaled = StandardScaler().fit_transform(X_immune)

# Run UMAP using only immune evasion genes to see if they alone can separate tumor stages
reducer_immune = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap_immune  = reducer_immune.fit_transform(X_immune_scaled)

plt.figure(figsize=(8, 6))
# Each dot is a patient, colored by their tumor stage
plt.scatter(X_umap_immune[:, 0], X_umap_immune[:, 1],
            c=[list(palette.values())[stage_order.index(s)] for s in stage_labels],
            s=60, alpha=0.8)
legend_elements = [Patch(facecolor=palette[s], label=s) for s in stage_order]
plt.legend(handles=legend_elements, title='Tumor Stage',
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("UMAP Projection of LUAD Samples\n(Immune Evasion Genes Only)")
plt.tight_layout()
plt.savefig('umap_immune.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# UMAP — Angiogenesis Genes Only
# Filter down to only the angiogenesis genes that were found in the dataset
angio_found = [g for g in angio_list if g in gene_list_found]
X_angio = LUAD_merged[angio_found].values
# Fill missing values and standardize just like before
X_angio = SimpleImputer(strategy='mean').fit_transform(X_angio)
X_angio_scaled = StandardScaler().fit_transform(X_angio)

# Run UMAP using only angiogenesis genes to see if they alone can separate tumor stages
reducer_angio = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap_angio  = reducer_angio.fit_transform(X_angio_scaled)

plt.figure(figsize=(8, 6))
# Each dot is a patient, colored by their tumor stage
plt.scatter(X_umap_angio[:, 0], X_umap_angio[:, 1],
            c=[list(palette.values())[stage_order.index(s)] for s in stage_labels],
            s=60, alpha=0.8)
legend_elements = [Patch(facecolor=palette[s], label=s) for s in stage_order]
plt.legend(handles=legend_elements, title='Tumor Stage',
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("UMAP Projection of LUAD Samples\n(Angiogenesis Genes Only)")
plt.tight_layout()
plt.savefig('umap_angio.png', dpi=150, bbox_inches='tight')
plt.show()