"""
Extended script incorporating sentiment-topic-emotion analysis concepts 
based on TDS model ideas from Akhmedov et al. (2021).

This:
- Loads data from Excel sheets.
- Performs topic modeling (LDA) over all activities.
- Conducts sentiment tagging on thought data.
- Correlates topics with emotion signals.
- Visualizes topic proportions alongside emotion trends.
- Analyzes alignment/divergence between expressed thoughts and detected emotions.
- Adds volatility visualizations and improved PCA for positive/negative word analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob  



def explicit_column_renaming(df):
    print("=== explicit_column_renaming Debugging ===")
    old_cols = df.columns.tolist()
    print("Original columns (after row-3 extraction):")
    print(old_cols, "\n")

    base_emos = [
        "Interest", "Boredom", "Happiness", "Anger",
        "Surprise", "Disappointment", "Satisfaction", "Confusion"
    ]
    new_cols = []

    for i, old_name in enumerate(old_cols):
        if i == 0:
            new_cols.append("Timestamp")
        elif i == 1:
            new_cols.append("Thought Data")
        elif i == 2:
            new_cols.append("Action Data")
        else:
            offset = i - 3
            chunk = offset // 8
            in_chunk = offset % 8
            if chunk == 0 and in_chunk < len(base_emos):
                new_cols.append(f"{base_emos[in_chunk]} (Facial)")
            elif chunk == 1 and in_chunk < len(base_emos):
                new_cols.append(f"{base_emos[in_chunk]} (Speech)")
            elif chunk == 2 and in_chunk < len(base_emos):
                new_cols.append(f"{base_emos[in_chunk]} (Language)")
            else:
                new_cols.append(old_name + "_EXTRA")

    df.columns = new_cols

    if df.columns.duplicated().any():
        print("  --> Detected duplicates. Making them unique with suffixes.")
        counts = {}
        final = []
        for c in df.columns:
            if c not in counts:
                counts[c] = 0
                final.append(c)
            else:
                counts[c] += 1
                final.append(f"{c}.{counts[c]}")
        df.columns = final

    final_cols = df.columns.tolist()
    print("Renaming done. Final columns =>")
    print(final_cols)
    print("=== End explicit_column_renaming Debugging ===\n")
    return df

#################
# 1)Reading the Excel

def read_and_unify_sheets(excel_file, sheets_of_focus):
    data_dict = {}
    try:
        xls = pd.ExcelFile(excel_file)
    except Exception as e:
        print(f"❌ Error loading Excel file '{excel_file}': {e}")
        return data_dict

    for sheet_name in sheets_of_focus:
        try:
            df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        except Exception as e:
            print(f"❌ Error reading sheet '{sheet_name}': {e}")
            continue

        if len(df_raw) < 4:
            print(f"⚠️  Sheet {sheet_name} has fewer than 4 rows => skipping.")
            continue

        col_names = df_raw.iloc[2].tolist()
        df_data = df_raw.iloc[3:].copy()
        df_data.columns = col_names

        df_data = explicit_column_renaming(df_data)

        if "Timestamp" in df_data.columns:
            df_data["Timestamp"] = pd.to_numeric(df_data["Timestamp"], errors='coerce')
            df_data.dropna(subset=["Timestamp"], inplace=True)
            df_data.sort_values("Timestamp", inplace=True)

        df_data.reset_index(drop=True, inplace=True)

        data_dict[sheet_name] = df_data
        print(f"✅ Completed sheet '{sheet_name}' => final shape {df_data.shape}\n")

    print("=== Done reading all sheets. ===\n")
    return data_dict

################################
# 2)Some Utilities
##############
def average_modalities(df, emotion):
    colnames = [
        f"{emotion} (Facial)",
        f"{emotion} (Speech)",
        f"{emotion} (Language)"
    ]
    found_cols = [c for c in colnames if c in df.columns]
    if not found_cols:
        return None
    tempdf = df[found_cols].apply(pd.to_numeric, errors='coerce')
    avg_ser = tempdf.mean(axis=1, skipna=True)
    return avg_ser

def get_3_peaks_and_3_troughs(ser):
    if not pd.api.types.is_numeric_dtype(ser):
        ser = pd.to_numeric(ser, errors='coerce')
    ser = ser.dropna()
    if len(ser) == 0:
        return [], []
    peak = ser.nlargest(3)
    trough = ser.nsmallest(3)
    return list(peak.index), list(trough.index)

##################################
# 3)The analyses (interest & boredom, correlation, distributions, PCA, etc.)
###########
def plot_interest_boredom(df, sheet_name):
    interest_ser = average_modalities(df, "Interest")
    boredom_ser = average_modalities(df, "Boredom")

    if interest_ser is None and boredom_ser is None:
        print(f"{sheet_name}: Skipping => no interest/boredom columns found.")
        return

    if "Timestamp" in df.columns:
        x_vals = df["Timestamp"].values
        x_label = "Time (seconds)"
    else:
        x_vals = np.arange(len(df))
        x_label = "Index"

    plt.figure(figsize=(10,5))
    if interest_ser is not None:
        plt.plot(x_vals, interest_ser, label="Interest (Avg Facial/Speech/Lang)", color='blue')
    if boredom_ser is not None:
        plt.plot(x_vals, boredom_ser, label="Boredom (Avg Facial/Speech/Lang)", color='red')
    plt.title(f"{sheet_name}: Interest & Boredom Over Time")
    plt.xlabel(x_label)
    plt.ylabel("Emotion Signal")
    plt.legend()
    plt.tight_layout()
    plt.show()

def do_correlation_matrices(df, sheet_name):
    facial_cols = [c for c in df.columns if "(Facial)" in c]
    lang_cols = [c for c in df.columns if "(Language)" in c]
    speech_cols = [c for c in df.columns if "(Speech)" in c]

    df_f_l = df[facial_cols + lang_cols].copy()
    for c in df_f_l.columns:
        df_f_l[c] = pd.to_numeric(df_f_l[c], errors='coerce')
    df_f_l.dropna(how='all', inplace=True)
    if df_f_l.shape[1] >= 2:
        corr_mat = df_f_l.corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".1f")
        plt.title(f"{sheet_name}: Correlation (Facial vs. Language)")
        plt.tight_layout()
        plt.show()
    else:
        print(f"{sheet_name}: Not enough numeric columns for Facial vs. Language correlation.")

    df_f_s = df[facial_cols + speech_cols].copy()
    for c in df_f_s.columns:
        df_f_s[c] = pd.to_numeric(df_f_s[c], errors='coerce')
    df_f_s.dropna(how='all', inplace=True)
    if df_f_s.shape[1] >= 2:
        corr_mat = df_f_s.corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".1f")
        plt.title(f"{sheet_name}: Correlation (Facial vs. Speech)")
        plt.tight_layout()
        plt.show()
    else:
        print(f"{sheet_name}: Not enough numeric columns for Facial vs. Speech correlation.")

import matplotlib.patches as mpatches  

import matplotlib.patches as mpatches  

def do_distribution_analysis_academic_vs_nonacademic(all_data):
    # Defining a color mapping for emotions (adjust colors as desired)
    emotion_colors = {
        "Interest": "#1f77b4",      # blue
        "Boredom": "#ff7f0e",       # orange
        "Happiness": "#2ca02c",     # green
        "Anger": "#d62728",         # red
        "Surprise": "#9467bd",      # purple
        "Disappointment": "#8c564b",# brown
        "Satisfaction": "#e377c2",  # pink
        "Confusion": "#7f7f7f"      # gray
    }

    acad_list = []
    nonacad_list = []
    for sh, df in all_data.items():
        if sh.startswith("A"):
            acad_list.append(df)
        elif sh.startswith("NA"):
            nonacad_list.append(df)

    if not acad_list or not nonacad_list:
        print("Skipping distribution analysis => missing academic or non-academic data.")
        return

    df_acad = pd.concat(acad_list, ignore_index=True)
    df_non = pd.concat(nonacad_list, ignore_index=True)

    #identify emotion columns and sort to group by modality
    all_emo_cols = [c for c in df_acad.columns if any(x in c for x in ["(Facial)", "(Speech)", "(Language)"])]
    sorted_cols = sorted(all_emo_cols, key=lambda x: (
        ("(Facial)" not in x),
        ("(Speech)" not in x),
        ("(Language)" not in x),
        x
    ))

    
    global_min = float('inf')
    global_max = float('-inf')
    for col in sorted_cols:
        combined = pd.concat([
            pd.to_numeric(df_acad[col], errors='coerce'),
            pd.to_numeric(df_non[col], errors='coerce')
        ]).dropna()
        if not combined.empty:
            global_min = min(global_min, combined.min())
            global_max = max(global_max, combined.max())

   
    if global_min == float('inf') or global_max == float('-inf'):
        global_min, global_max = 0, 1

    cols_per_row = 3
    num_plots = len(sorted_cols)
    rows = math.ceil(num_plots / cols_per_row)
    fig_width = 7
    fig_height = 1.5 * rows

 
    def create_legend_patches():
        return [mpatches.Patch(color=color, label=emo) for emo, color in emotion_colors.items()]

    #academic distribution analysis
    print("=== Distribution Analysis: Academic ===")
    fig, axs = plt.subplots(rows, cols_per_row, figsize=(fig_width, fig_height))
    axs = axs.flatten() if num_plots > 1 else [axs]

    for i, col in enumerate(sorted_cols):
        if i >= len(axs):
            break
        sub_ser = pd.to_numeric(df_acad[col], errors='coerce').dropna()
        ax = axs[i]
       
        emotion = col.split(" (")[0]
        color = emotion_colors.get(emotion, 'blue')

        if len(sub_ser) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=5)
            ax.set_title(col, fontsize=6)
        else:
            sns.histplot(sub_ser, ax=ax, bins=20, kde=True, color=color)
            ax.set_title(f"{col}", fontsize=6)

        ax.set_xlabel("Value", fontsize=5)
        ax.set_ylabel("Freq.", fontsize=5)
        ax.tick_params(axis='both', which='major', labelsize=4)
        ax.set_xlim(global_min, global_max)  # Set consistent x-axis limits

    for j in range(i+1, len(axs)):
        axs[j].set_visible(False)

    # Add legend
    legend_patches = create_legend_patches()
    fig.legend(handles=legend_patches, loc='upper right', fontsize=6, title="Emotions")

    plt.suptitle("Academic Emotion Distributions", fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.show()

    #Non-Academic distribution analysis
    print("=== Distribution Analysis: Non-Academic ===")
    fig, axs = plt.subplots(rows, cols_per_row, figsize=(fig_width, fig_height))
    axs = axs.flatten() if num_plots > 1 else [axs]

    for i, col in enumerate(sorted_cols):
        if i >= len(axs):
            break
        sub_ser = pd.to_numeric(df_non[col], errors='coerce').dropna()
        ax = axs[i]
        emotion = col.split(" (")[0]
        color = emotion_colors.get(emotion, 'green')

        if len(sub_ser) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=5)
            ax.set_title(col, fontsize=6)
        else:
            sns.histplot(sub_ser, ax=ax, bins=20, kde=True, color=color)
            ax.set_title(f"{col}", fontsize=6)

        ax.set_xlabel("Value", fontsize=5)
        ax.set_ylabel("Freq.", fontsize=5)
        ax.tick_params(axis='both', which='major', labelsize=4)
        ax.set_xlim(global_min, global_max)  #Use same x-axis limits

    for j in range(i+1, len(axs)):
        axs[j].set_visible(False)

   
    legend_patches = create_legend_patches()
    fig.legend(handles=legend_patches, loc='upper right', fontsize=6, title="Emotions")

    plt.suptitle("Non-Academic Emotion Distributions", fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.show()


####################
# 4) PCA with top-50 function words for 
########
def get_top_50_function_words_thoughts(df_list):
    """
    Updated function to use the FULL list of function words 
    that you provided (instead of the smaller hard-coded set).
    """
    candidate_funcs = {
        "the","an","a","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve",
        "thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty","hundred","hundreds",
        "thousand","thousands","million","millions","billion","billions","first","second","third","fourth","fifth",
        "sixth","seventh","eighth","ninth","tenth","eleventh","twelfth","thirteenth","fourteenth","fifteenth",
        "sixteenth","seventeenth","eighteenth","nineteenth","twentieth","i","you","he","she","it","we","they",
        "me","him","us","them","this","that","these","those","my","your","his","her","its","their","our","mine",
        "yours","hers","theirs","ours","all","some","many","lot","lots","ton","tons","bit","no","every","enough",
        "little","much","more","most","plenty","several","few","fewer","kind","kinds","myself","yourself","himself",
        "herself","itself","oneself","ourselves","yourselves","themselves","none","nobody","everyone","everybody",
        "someone","somebody","anyone","anybody","nothing","everything","something","anything","each","other",
        "whatever","whichever","whoever","whomever","whomsoever","whosoever","others","neither","both","either","any",
        "such","one's","nobody's","everyone's","everybody's","someone's","somebody's","anyone's","anybody's",
        "nothing's","everything's","something's","anything's","whoever's","others'","other's","another's","neither's",
        "either's","which","what","whose","who","whom","where","how","why","whether","wherever","whyever",
        "wheresoever","whensoever","howsoever","whysoever","whatsoever","whereso","whomso","whenso","howso","whyso",
        "whoso","whatso","therefor","therein","hereby","hereto","wherein","therewith","herewith","wherewith",
        "thereby","there","here","whither","thither","hither","whence","thence","always","once","twice","thrice",
        "can","cannot","can't","could","couldn't","could've","dare","dares","dared","do","don't","does","doesn't",
        "did","didn't","done","have","haven't","had","hadn't","has","hasn't","i've","you've","we've","they've",
        "i'd","you'd","he'd","she'd","it'd","we'd","they'd","would","wouldn't","would've","may","might","must",
        "need","needn't","needs","ought","shall","shalln't","shan't","should","shouldn't","will","won't","i'll",
        "you'll","he'll","she'll","it'll","we'll","they'll","there's","there're","there'll","here's","here're",
        "appear","appears","appeared","become","becomes","became","come","comes","came","keep","keeps","kept",
        "remain","remains","remained","stay","stays","stayed","turn","turns","turned","doing","daring","having",
        "appearing","becoming","coming","keeping","remaining","staying","saying","asking","stating","seeming",
        "letting","making","setting","showing","putting","adding","going","using","trying","containing","in","from",
        "with","under","throughout","atop","for","on","of","to","aboard","about","above","abreast","absent","across",
        "adjacent","after","against","along","alongside","amid","mid","among","apropos","apud","around","as","astride",
        "at","ontop","afore","tofore","behind","ahind","below","ablow","beneath","neath","beside","between","atween",
        "beyond","ayond","by","chez","circa","spite","down","except","into","less","like","minus","near","nearer",
        "nearest","anear","notwithstanding","off","onto","opposite","out","outen","over","past","per","pre","qua",
        "sans","sauf","sithence","through","thru","truout","toward","underneath","up","upon","upside","versus","via",
        "vis-à-vis","without","ago","apart","aside","aslant","away","withal","towards","amidst","amongst","midst",
        "whilst","back","within","forward","backward","ahead","and","or","and/or","yet","sooner","just","only","if",
        "even","say","says","said","claimed","ask","asks","asked","stated","explain","explains","explained","think",
        "thinks","talks","talked","announces","announced","tells","told","discusses","discussed","suggests","suggested",
        "understands","understood","again","definitely","eternally","expressively","instead","expressly","immediately",
        "including","instantly","namely","naturally","next","notably","now","nowadays","ordinarily","positively",
        "truly","ultimately","uniquely","usually","almost","maybe","probably","granted","initially","too","actually",
        "already","e.g","i.e","often","regularly","simply","optionally","perhaps","sometimes","likely","never","ever",
        "else","inasmuch","provided","currently","incidentally","elsewhere","particular","recently","relatively","f.i",
        "clearly","apparently","highly","very","really","extremely","absolutely","completely","totally","utterly",
        "quite","somewhat","seriously","fairly","fully","amazingly","seem","seems","seemed","let","let's","lets",
        "make","makes","made","want","showed","shown","go","goes","went","gone","take","takes","took","taken","put",
        "puts","use","used","try","tries","tried","mean","means","meant","called","based","add","adds","added",
        "contain","contains","contained","consist","consists","consisted","ensure","ensures","ensured","new","newer",
        "newest","old","older","oldest","previous","good","well","better","best","big","bigger","biggest","easy",
        "easier","easiest","fast","faster","fastest","far","hard","harder","hardest","least","own","large","larger",
        "largest","long","longer","longest","low","lower","lowest","high","higher","highest","regular","simple",
        "simpler","simplest","small","smaller","smallest","tiny","tinier","tiniest","short","shorter","shortest",
        "main","actual","nice","nicer","nicest","real","same","able","certain","usual","so-called","mainly","mostly",
        "recent","anymore","complete","lately","possible","commonly","constantly","continually","directly","easily",
        "nearly","slightly","somewhere","estimated","latest","different","similar","widely","bad","worse","worst",
        "great","specific","available","average","awful","awesome","basic","beautiful","busy","current","entire",
        "everywhere","important","major","multiple","normal","necessary","obvious","partly","special","last","early",
        "earlier","earliest","young","younger","youngest","oh","wow","tut-tut","tsk-tsk","ugh","whew","phew","yeah",
        "yea","shh","oops","ouch","aha","yikes","tbs","tbsp","spk","lb","qt","pk","bu","oz","pt","mod","doz","hr",
        "f.g","ml","dl","cl","l","mg","g","kg","quart","seconds","minute","minutes","hour","hours","day","days",
        "week","weeks","month","months","year","years","today","tomorrow","yesterday","thing","things","way","ways",
        "matter","case","likelihood","ones","piece","pieces","stuff","times","part","parts","percent","instance",
        "instances","aspect","aspects","item","items","idea","theme","person","detail","details","factor","factors",
        "difference","differences","not","yes","sure","top","bottom","ok","okay","amen","aka","etc","etcetera",
        "sorry","please","ms","mss","mrs","mr","dr","prof","jr","sr.","i'm"
    }

    freq = {}
    for df in df_list:
        if "Thought Data" not in df.columns:
            continue
        for txt in df["Thought Data"].dropna():
            words = str(txt).lower().split()
            for w in words:
                w = "".join(ch for ch in w if ch.isalpha())
                if w in candidate_funcs:
                    freq[w] = freq.get(w, 0) + 1

    sorted_pairs = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    top_50 = [x[0] for x in sorted_pairs[:50]]
    return top_50

####################################
# 4) PCA with top-50 function words for (Academic vs. Non-Academic)
############
def do_pca_academic_vs_nonacademic_thoughts(data_dict):
    acad = []
    nonacad = []
    for sh, df in data_dict.items():
        if sh.startswith("A"):
            acad.append(df)
        elif sh.startswith("NA"):
            nonacad.append(df)

    if not acad and not nonacad:
        print("No data for PCA (Academic vs Non).")
        return

    topA = get_top_50_function_words_thoughts(acad)
    topN = get_top_50_function_words_thoughts(nonacad)
    union_top = list(set(topA + topN))
    if len(union_top) > 50:
        union_top = union_top[:50]

    combined_data = []
    domain_label = []

    for df in acad:
        if "Thought Data" not in df.columns:
            continue
        for idx in range(len(df)):
            combined_data.append(df.iloc[idx])
            domain_label.append("Academic")

    for df in nonacad:
        if "Thought Data" not in df.columns:
            continue
        for idx in range(len(df)):
            combined_data.append(df.iloc[idx])
            domain_label.append("Non-Academic")

    if not combined_data:
        print("No text rows for PCA.")
        return

    from collections import Counter
    mat = []
    for row in combined_data:
        txt = str(row.get("Thought Data", "")).lower()
        words = txt.split()
        words = ["".join(ch for ch in w if ch.isalpha()) for w in words]
        c = Counter(words)
        vec = [c[w] for w in union_top]
        mat.append(vec)

    mat = np.array(mat, dtype=float)
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        print("Cannot do PCA => matrix too small.")
        return

    pca = PCA(n_components=2)
    coords = pca.fit_transform(mat)
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100

    plt.figure(figsize=(8, 6))


    for i in range(len(coords)):
        if domain_label[i] == "Academic":
            plt.scatter(coords[i, 0], coords[i, 1], color='blue', s=10)
        else:
            plt.scatter(coords[i, 0], coords[i, 1], color='red', s=10)

    # Draw quadrant lines
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    acad_patch = plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='blue', label='Academic', markersize=5)
    non_patch = plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor='red', label='Non-Academic', markersize=5)
    plt.legend(handles=[acad_patch, non_patch], loc='best')
    plt.xlabel(f"PC1 ({pc1_var:.2f}% explained)")
    plt.ylabel(f"PC2 ({pc2_var:.2f}% explained)")
    plt.title("Principal Component Analysis of Top-50 Function Words: Academic vs Non-Academic")
    plt.tight_layout()
    plt.show()


#####
# 5) PCA for Activity 3 vs. Non-Activity 3 with 50 pos/neg words (Improved)
########################
def improved_posneg_pca_activity3_vs_nonacad3(data_dict):

    acad3_key = None
    non3_key = None
    for k in data_dict.keys():
        if k.startswith("A3"):
            acad3_key = k
        elif k.startswith("NA3"):
            non3_key = k

    if not acad3_key or not non3_key:
        print("Cannot do pos/neg PCA => missing 'A3' or 'NA3' in sheet keys.")
        return

    df_a3 = data_dict[acad3_key]
    df_n3 = data_dict[non3_key]

    # Define emotion groups
    positive_emotions = ["Interest", "Happiness", "Surprise", "Disappointment", "Satisfaction"]
    negative_emotions = ["Boredom", "Anger", "Disappointment", "Confusion"]

    
    def avg_emotion_threshold(df, emotions):
        scores = []
        for emo in emotions:
            ser = average_modalities(df, emo)
            if ser is not None:
                scores.append(ser)
        if scores:
            combined = pd.concat(scores, axis=1)
            return combined.mean(axis=1).mean()
        return None

    pos_threshold_a3 = avg_emotion_threshold(df_a3, positive_emotions)
    neg_threshold_a3 = avg_emotion_threshold(df_a3, negative_emotions)


    def extract_words_by_emotion_threshold(df, emotions, threshold):
        extracted_words = []
        for _, row in df.iterrows():
            values = []
            for emo in emotions:
                for modality in ["(Facial)", "(Speech)", "(Language)"]:
                    col = f"{emo} {modality}"
                    if col in row and pd.notna(row[col]):
                        try:
                            values.append(float(row[col]))
                        except:
                            pass
     
            if values and np.mean(values) > threshold:
                if "Thought Data" in row and isinstance(row["Thought Data"], str):
                    words = row["Thought Data"].lower().split()
                    words = ["".join(ch for ch in w if ch.isalpha()) for w in words if w]
                    extracted_words.extend(words)
        return extracted_words

   
    positive_words_a3 = extract_words_by_emotion_threshold(df_a3, positive_emotions, pos_threshold_a3) if pos_threshold_a3 is not None else []
    negative_words_a3 = extract_words_by_emotion_threshold(df_a3, negative_emotions, neg_threshold_a3) if neg_threshold_a3 is not None else []
    positive_words_n3 = extract_words_by_emotion_threshold(df_n3, positive_emotions, pos_threshold_a3) if pos_threshold_a3 is not None else []
    negative_words_n3 = extract_words_by_emotion_threshold(df_n3, negative_emotions, neg_threshold_a3) if neg_threshold_a3 is not None else []

    from collections import Counter
    combined_positive = positive_words_a3 + positive_words_n3
    combined_negative = negative_words_a3 + negative_words_n3

    pos_counter = Counter(combined_positive)
    neg_counter = Counter(combined_negative)

    top_positive = [word for word, _ in pos_counter.most_common(50)]
    top_negative = [word for word, _ in neg_counter.most_common(50)]


    wordlist = list(set(top_positive + top_negative))
    if not wordlist:
        print("No significant words found based on emotion thresholds.")
        return

    def vectorize(df, wordlist):
        mat_ = []
        for i in range(len(df)):
            txt = str(df.loc[i, "Thought Data"]) if "Thought Data" in df.columns else ""
            words = txt.lower().split()
            words = ["".join(ch for ch in w if ch.isalpha()) for w in words]
            counts = Counter(words)
            rowvec = [counts[w] for w in wordlist]
            mat_.append(rowvec)
        return np.array(mat_, dtype=float)

    matA = vectorize(df_a3, wordlist)
    matN = vectorize(df_n3, wordlist)
    if matA.size == 0 or matN.size == 0:
        print("No Thought Data in one of the sheets => skipping improved pos/neg PCA for A3 vs NA3.")
        return

    big_mat = np.vstack([matA, matN])
    labels = ["A3"] * len(matA) + ["NA3"] * len(matN)

    if big_mat.shape[0] < 2 or big_mat.shape[1] < 2:
        print("Matrix too small for PCA => skipping.")
        return

    pca = PCA(n_components=2)
    coords = pca.fit_transform(big_mat)
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100

    plt.figure(figsize=(8, 6))
    for i in range(len(coords)):
        if labels[i] == "A3":
            plt.scatter(coords[i, 0], coords[i, 1], c='purple', s=15)
        else:
            plt.scatter(coords[i, 0], coords[i, 1], c='orange', s=15)


    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    legend_elts = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple',
                   label='Academic Activity 3', markersize=6),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   label='Non-Academic Activity 3', markersize=6)
    ]
    plt.legend(handles=legend_elts)
    plt.xlabel(f"PC1: pos/neg words ({pc1_var:.2f}% explained)")
    plt.ylabel(f"PC2: pos/neg words ({pc2_var:.2f}% explained)")
    plt.title("Principal Component Analysis of Emotion-Driven Word Usage: : A3 vs NA3")
    plt.tight_layout()
    plt.show()



def lda_over_all_sheets(data_dict, n_topics=5):
    #Combine all Thought Data across sheets
    all_texts = []
    for df in data_dict.values():
        if "Thought Data" in df.columns:
            all_texts.extend(df["Thought Data"].dropna().astype(str).tolist())
    if not all_texts:
        print("No Thought Data found across sheets.")
        return

    #perform LDA on combined texts
    vectorizer = CountVectorizer(
        max_features=500, 
        stop_words='english',
        token_pattern=r"(?u)\b\w[\w']+\b"
    )
    X = vectorizer.fit_transform(all_texts)
    if X.shape[0] < n_topics:
        print("Not enough documents for global LDA.")
        return

    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=999)
    lda_model.fit(X)
    feats = vectorizer.get_feature_names_out()

    print("--- Global LDA Topics ---")
    for topic_idx, topic in enumerate(lda_model.components_):
        topidx = topic.argsort()[::-1][:9]
        topwords = [feats[i] for i in topidx]
        print(f"Topic {topic_idx}: {topwords}")

def volatility_bar_charts(df, sheet_name):
    #Calculate volatility (standard deviation) for each emotion in the sheet and plot as bar chart
    emotions = ["Interest", "Boredom", "Happiness", "Anger", 
                "Surprise", "Disappointment", "Satisfaction", "Confusion"]
    volatilities = {}
    for emo in emotions:
        ser = average_modalities(df, emo)
        if ser is not None:
            #comput standard deviation as volatility
            volatilities[emo] = np.std(ser.dropna())
    if not volatilities:
        print(f"No emotion data for volatility bar chart in {sheet_name}.")
        return

    plt.figure(figsize=(10,5))
    sns.barplot(x=list(volatilities.keys()), y=list(volatilities.values()))
    plt.title(f"{sheet_name}: Emotion Volatility (σ)")
    plt.ylabel("Standard Deviation (σ)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def academic_vs_nonacademic_volatility_line_chart(data_dict):
  
    academics = [df for sh, df in data_dict.items() if sh.startswith("A")]
    nonacademics = [df for sh, df in data_dict.items() if sh.startswith("NA")]

    if not academics or not nonacademics:
        print("Insufficient data for volatility line chart.")
        return

    
    acad_min_timestamps = [df["Timestamp"].min() for df in academics if "Timestamp" in df.columns]
    if not acad_min_timestamps:
        print("No timestamps found in academic activities.")
        return
    overall_min_timestamp = max(acad_min_timestamps)  

    #
    def filter_and_align(df_list, min_timestamp):
        aligned_vol = []
        for df in df_list:
            if "Timestamp" not in df.columns:
                continue
            # Filter rows up to the min_timestamp
            filtered_df = df[df["Timestamp"] <= min_timestamp]
            if filtered_df.empty:
                continue
            row_stds = []
            for emo in ["Interest", "Boredom", "Happiness", "Anger", 
                        "Surprise", "Disappointment", "Satisfaction", "Confusion"]:
                ser = average_modalities(filtered_df, emo)
                if ser is not None:
                    row_stds.append(np.std(ser.dropna()))
            if row_stds:
                aligned_vol.append(np.mean(row_stds))
        return aligned_vol

    acad_vol = filter_and_align(academics, overall_min_timestamp)
    nonacad_vol = filter_and_align(nonacademics, overall_min_timestamp)

    
    x_acad = np.arange(len(acad_vol))
    x_nonacad = np.arange(len(nonacad_vol))

    plt.figure(figsize=(10,5))
    plt.plot(x_acad, acad_vol, label='Academic Volatility (σ)', marker='o')
    plt.plot(x_nonacad, nonacad_vol, label='Non-Academic Volatility (σ)', marker='x')
    plt.title("Volatility of Emotions: Academic vs Non-Academic")
    plt.xlabel("Document Index (Aligned to Min Academic Timestamp)")
    plt.ylabel("Average Standard Deviation (σ)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def quick_lda_with_sentiment(df, sheet_name, n_topics=5):
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    if "Thought Data" not in df.columns:
        print(f"{sheet_name}: No 'Thought Data' => skipping LDA.")
        return

    texts = df["Thought Data"].fillna("").astype(str).tolist()
    if not any(len(t.strip()) > 0 for t in texts):
        print(f"{sheet_name}: All empty => skipping LDA.")
        return

    vectorizer = CountVectorizer(
        max_features=500, 
        stop_words='english',
        token_pattern=r"(?u)\b\w[\w']+\b"
    )
    X = vectorizer.fit_transform(texts)
    if X.shape[0] < 5:
        print(f"{sheet_name}: Not enough docs for LDA => skip.")
        return

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=999)
    lda.fit(X)
    topics_distribution = lda.transform(X)
    feats = vectorizer.get_feature_names_out()

    print(f"--- {sheet_name}: LDA Topics ---")
    topic_keywords = {}
    for i, comp in enumerate(lda.components_):
        topidx = np.argsort(comp)[::-1][:10]
        topwords = [feats[x] for x in topidx]
        topic_keywords[f"Topic_{i}"] = topwords
        print(f"Topic {i}: {topwords}")

    topic_cols = [f"Topic_{i}" for i in range(n_topics)]
    df_topics = pd.DataFrame(topics_distribution, columns=topic_cols, index=df.index)

    
    sentiment_scores = []
    for text in texts:
        blob = TextBlob(text)
        sentiment_scores.append(blob.sentiment.polarity)
    df_topics['SentimentScore'] = sentiment_scores

    emotions_to_check = ["Interest", "Boredom", "Frustration", "Happiness", "Anger"]
    correlation_results = {}
    for emotion in emotions_to_check:
        emo_series = average_modalities(df, emotion)
        if emo_series is None:
            continue
        correlation_results[emotion] = {}
        for tcol in topic_cols:
            common_idx = df_topics[tcol].dropna().index.intersection(emo_series.dropna().index)
            if common_idx.empty:
                continue
            topic_vals = df_topics.loc[common_idx, tcol]
            emo_vals = emo_series.loc[common_idx]
            if len(topic_vals) < 2:
                continue
            corr = np.corrcoef(topic_vals, emo_vals)[0, 1]
            correlation_results[emotion][tcol] = corr
            print(f"Correlation between {tcol} and {emotion}: {corr:.2f}")

    frustration_keywords = {"frustration", "angry", "irritated", "annoyed", "difficult", "challenging"}
    for topic, keywords in topic_keywords.items():
        if any(word in keywords for word in frustration_keywords):
            for emotion, topic_corrs in correlation_results.items():
                if emotion == "Interest" and topic in topic_corrs:
                    corr_val = topic_corrs[topic]
                    print(f"Topic {topic} (frustration-related) correlation with Interest: {corr_val:.2f}")
                    if corr_val > 0:
                        print(f"  -> Divergence: Frustration-related thoughts coincide with rising Interest.")
                    else:
                        print(f"  -> Alignment: Frustration-related thoughts align with lower Interest.")

    print(f"--- End of extended LDA analysis for {sheet_name} ---\n")
    return df_topics, topic_keywords

def dummy_confusion_matrix(df, sheet_name):
    if len(df) < 6:
        return

    if "GroundTruthEmotion" not in df.columns or "PredictedEmotion" not in df.columns:
        emos = ["Happy", "Sad", "Confused"]
        import random
        glist, plist = [], []
        for _ in range(len(df)):
            glist.append(random.choice(emos))
            plist.append(random.choice(emos))
        df["GroundTruthEmotion"] = glist
        df["PredictedEmotion"] = plist

    from sklearn.metrics import confusion_matrix, classification_report
    valid = df.dropna(subset=["GroundTruthEmotion", "PredictedEmotion"])
    if len(valid) == 0:
        return

    ytrue = valid["GroundTruthEmotion"]
    ypred = valid["PredictedEmotion"]
    labs = sorted(list(set(ytrue) | set(ypred)))
    cmat = confusion_matrix(ytrue, ypred, labels=labs)

    print(f"\n=== {sheet_name} Dummy Confusion Matrix ===")
    print("Labels order:", labs)
    import numpy as np
    print(np.around(cmat.astype(float), 1))
    
    print("Classification Report =>")
    print(classification_report(ytrue, ypred, labels=labs, digits=1))

#####
# New Methods for Function Extraction and Extended LDA Emotion Analysis
####################################
def extract_top_function_words(all_data, top_n=300):
    all_thoughts = []
    for df in all_data.values():
        if "Thought Data" in df.columns:
            all_thoughts.extend(df["Thought Data"].dropna().astype(str).tolist())
    words = []
    for text in all_thoughts:
        for word in text.lower().split():
            cleaned = ''.join(ch for ch in word if ch.isalpha())
            if cleaned:
                words.append(cleaned)
    from collections import Counter
    freq = Counter(words)
    top_words = [word for word, _ in freq.most_common(top_n)]
    return set(top_words)

def lda_emotion_pilot_plot_extended(df, sheet_name, function_words, n_topics=5, window_size=5):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer


    if "Thought Data" not in df.columns:
        print(f"{sheet_name}: No 'Thought Data' => skipping extended LDA plot.")
        return

    texts = df["Thought Data"].fillna("").astype(str).tolist()
    if not any(len(t.strip()) > 0 for t in texts):
        print(f"{sheet_name}: All empty => skipping extended LDA plot.")
        return

    #Vectorization and LDA with custom token pattern
    vectorizer = CountVectorizer(
        max_features=500, 
        stop_words=list(function_words),
        token_pattern=r"(?u)\b\w[\w']+\b"
    )
    X = vectorizer.fit_transform(texts)
    if X.shape[0] < n_topics:
        print(f"{sheet_name}: Not enough documents for {n_topics} topics => skipping LDA plot.")
        return

    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=999)
    lda_model.fit(X)
    topic_distributions = lda_model.transform(X)
    feats = vectorizer.get_feature_names_out()

    print(f"--- {sheet_name}: Topics for Extended Plot ---")
    topic_keywords = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        topidx = topic.argsort()[::-1][:10]
        topwords = [feats[i] for i in topidx]
        topic_keywords[topic_idx] = topwords
        print(f"Topic {topic_idx}: {topwords}")

    topic_variances = np.var(topic_distributions, axis=0)
    topic_variances = np.round(topic_variances, 4)

    
    selected_topics = np.argsort(topic_variances)[-2:]


    topic_info = []
    for idx in range(n_topics):
        topic_info.append({
            "Topic": f"Topic {idx}",
            "Top Words": ", ".join(topic_keywords[idx]),
            "Variance": topic_variances[idx]
        })
    df_topic_info = pd.DataFrame(topic_info)


    df_topic_info_sorted = df_topic_info.sort_values(by="Variance", ascending=False)

    fig_table, ax_table = plt.subplots(figsize=(12, 0.5 + 0.4 * n_topics))  
    ax_table.axis('tight')
    ax_table.axis('off')
    table = ax_table.table(cellText=df_topic_info_sorted.values,
                           colLabels=df_topic_info_sorted.columns,
                           cellLoc='left',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5) 
    plt.title(f"{sheet_name}: Topic Variance Table")
    plt.tight_layout()
    plt.show()

    print("\nTwo topics with highest variance:")
    for topic_idx in selected_topics:
        print(f"Topic {topic_idx} - Variance: {topic_variances[topic_idx]:.4f}, Top Words: {', '.join(topic_keywords[topic_idx])}")

    #
    interest_signal = average_modalities(df, "Interest")
    boredom_signal = average_modalities(df, "Boredom")

    n_docs = topic_distributions.shape[0]
    interest_vals = interest_signal.values[:n_docs] if interest_signal is not None else np.full(n_docs, np.nan)
    boredom_vals = boredom_signal.values[:n_docs] if boredom_signal is not None else np.full(n_docs, np.nan)

 
    def moving_average(data, w):
        return np.convolve(data, np.ones(w) / w, mode='same')


    if "Timestamp" in df.columns:
        x = df["Timestamp"].values[:n_docs]
        x_label = "Time (seconds)"
    else:
        x = np.arange(n_docs)
        x_label = "Document Index"


    smoothed_topics = []
    for topic_idx in selected_topics:
        raw_series = topic_distributions[:, topic_idx]
        smoothed_series = moving_average(raw_series, window_size)
        smoothed_topics.append(smoothed_series)


    fig, ax1 = plt.subplots(figsize=(12, 6))
  
    ax1.stackplot(x, smoothed_topics,
                  labels=[f"Topic {selected_topics[0]}", f"Topic {selected_topics[1]}"],
                  alpha=0.5)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Topic Proportion (Stacked)")
    ax1.legend(loc='upper left')

    #Add twin axis for emotion signals
    ax2 = ax1.twinx()
    if interest_signal is not None:
        ax2.plot(x, interest_vals, color='blue', linestyle='--', label='Interest')
    if boredom_signal is not None:
        ax2.plot(x, boredom_vals, color='red', linestyle='--', label='Boredom')
    ax2.set_ylabel("Emotion Signal")
    ax2.legend(loc='upper right')

    plt.title(f"{sheet_name}: Stacked Topic Proportions and Emotion Trends Over Time")
    plt.tight_layout()
    plt.show()

def export_figure_list_to_doc(fig_list, doc_filename="Plot_Figures.doc"):

    from collections import OrderedDict

    sheet_to_files = OrderedDict()
    figure_counter = 1

    for (sheet, fname) in fig_list:
        if sheet not in sheet_to_files:
            sheet_to_files[sheet] = []
        sheet_to_files[sheet].append(fname)

    with open(doc_filename, "w", encoding="utf-8") as f:
        for sheet, file_list in sheet_to_files.items():
            # Write a heading or label for this sheet
            f.write(f"=== {sheet} ===\n")
            for fname in file_list:
                f.write(f"Figure{figure_counter}: {fname}\n")
                figure_counter += 1
            f.write("\n")  #blank line between sheets


# 7) MAIN
####################################
def main():
    excel_file = "Human AI Record Template_V16.xlsx"  
    sheets_of_focus = [
        "A1 - Cellular Automata Model",
        "A2 - MP Birthplace Analysis",
        "A3 - Deep Learning Model",
        "NA1 - Internship Preparation",
        "NA2 - Christmas Gift Selection",
        "NA3 - Event Coordination"
    ]
    data_dict = read_and_unify_sheets(excel_file, sheets_of_focus)

    
    lda_over_all_sheets(data_dict)


    for sheet_name, df in data_dict.items():
        plot_interest_boredom(df, sheet_name)
        do_correlation_matrices(df, sheet_name)
        dummy_confusion_matrix(df, sheet_name)
        quick_lda_with_sentiment(df, sheet_name, n_topics=5)
        volatility_bar_charts(df, sheet_name)  

    do_distribution_analysis_academic_vs_nonacademic(data_dict)
    do_pca_academic_vs_nonacademic_thoughts(data_dict)
    improved_posneg_pca_activity3_vs_nonacad3(data_dict)  

    academic_vs_nonacademic_volatility_line_chart(data_dict)  

    #extract top function words and perform extended LDA plots for all sheets
    top_function_words = extract_top_function_words(data_dict, top_n=100)
    print("Top function words extracted:", top_function_words)
    for sheet_name, df in data_dict.items():
        lda_emotion_pilot_plot_extended(
            df, sheet_name, 
            function_words=top_function_words, n_topics=5
        )

    print("=== All analysis done. ===")
    export_figure_list_to_doc(FIGURE_LOG, "Plot_Figures.doc")

if __name__=="__main__":
    main()
