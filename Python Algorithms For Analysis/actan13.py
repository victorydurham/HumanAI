

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

######################
# GLOBAL:Track figure filenames & associated sheet
FIGURE_LOG = []  #Stores tuples: (sheet_name, filename)

def finalize_plot_and_save(sheet_name, chart_title, dpi=150):
 
    plt.title(chart_title, pad=5)
    plt.tight_layout()

    safe_title = re.sub(r'[\\/*?:"<>|]', '_', chart_title)
    safe_title = safe_title.replace(' ', '_')
    filename = f"{safe_title}.png"
    plt.savefig(filename, dpi=dpi)
    plt.show()


    global FIGURE_LOG
    FIGURE_LOG.append((sheet_name, filename))

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
            #heading or label for this sheet
            f.write(f"=== {sheet} ===\n")
            for fname in file_list:
                f.write(f"Figure{figure_counter}: {fname}\n")
                figure_counter += 1
            f.write("\n")  # blank line between sheets


####################
#Basic Utilities
#######
def explicit_column_renaming_tds(df):
    old_cols = df.columns.tolist()
    base_emos = [
        "Interest","Boredom","Happiness","Anger",
        "Surprise","Disappointment","Satisfaction","Confusion"
    ]
    new_cols=[]
    for i,col in enumerate(old_cols):
        if i==0:
            new_cols.append("Timestamp")
        elif i==1:
            new_cols.append("Thought Data")
        elif i==2:
            new_cols.append("Action Data")
        else:
            offset=i-3
            chunk=offset//8
            in_chunk=offset%8
            if chunk==0 and in_chunk<len(base_emos):
                new_cols.append(f"{base_emos[in_chunk]} (Facial)")
            elif chunk==1 and in_chunk<len(base_emos):
                new_cols.append(f"{base_emos[in_chunk]} (Speech)")
            elif chunk==2 and in_chunk<len(base_emos):
                new_cols.append(f"{base_emos[in_chunk]} (Language)")
            else:
                new_cols.append(col+"_EXTRA")
    df.columns=new_cols

 
    if df.columns.duplicated().any():
        counts={}
        final=[]
        for c in df.columns:
            if c not in counts:
                counts[c]=0
                final.append(c)
            else:
                counts[c]+=1
                final.append(f"{c}.{counts[c]}")
        df.columns=final
    return df

def read_and_unify_sheets_tds(excel_file, sheets_of_focus):
  
    data_dict={}
    if not os.path.isfile(excel_file):
        print(f"File not found: {excel_file}")
        return data_dict

    xls = pd.ExcelFile(excel_file)
    for sheet_name in sheets_of_focus:
        try:
            df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        except Exception as e:
            print(f"Error reading '{sheet_name}': {e}")
            continue
        if len(df_raw)<4:
            continue
        col_names=df_raw.iloc[2].tolist()
        df_data=df_raw.iloc[3:].copy()
        df_data.columns=col_names
        df_data=explicit_column_renaming_tds(df_data)

        if "Timestamp" in df_data.columns:
            df_data["Timestamp"]=pd.to_numeric(df_data["Timestamp"], errors='coerce')
            df_data.dropna(subset=["Timestamp"], inplace=True)
            df_data.sort_values("Timestamp", inplace=True)

        df_data.reset_index(drop=True, inplace=True)
        data_dict[sheet_name]=df_data

    return data_dict

def average_modalities(df,emotion):
    """
    Averages across (Facial, Speech, Language) columns for the given emotion 
    if they exist. Returns a Series or None if not found.
    """
    cands = [f"{emotion} (Facial)", f"{emotion} (Speech)", f"{emotion} (Language)"]
    f = [c for c in cands if c in df.columns]
    if not f:
        return None
    tmp = df[f].apply(pd.to_numeric, errors='coerce')
    return tmp.mean(axis=1,skipna=True)


# interest/boredom, correlation, LDA, volatility
##############
def plot_interest_boredom(df,sheet_name):
    interest=average_modalities(df,"Interest")
    boredom=average_modalities(df,"Boredom")
    if interest is None and boredom is None:
        return
    if "Timestamp" in df.columns:
        x_vals=df["Timestamp"].values
        x_label="Timestamp"
    else:
        x_vals=np.arange(len(df))
        x_label="Index"

    plt.figure(figsize=(10,5))
    if interest is not None:
        plt.plot(x_vals,interest,label="Interest(avg)",color='blue')
    if boredom is not None:
        plt.plot(x_vals,boredom,label="Boredom(avg)",color='red')
    chart_title = f"{sheet_name}: Interest & Boredom Over Time"
    plt.xlabel(x_label)
    plt.ylabel("Emotion Signal")
    plt.legend()
    finalize_plot_and_save(sheet_name, chart_title)

def plot_interest_confusion(df,sheet_name):
    interest=average_modalities(df,"Interest")
    confusion=average_modalities(df,"Confusion")
    if interest is None and confusion is None:
        return
    if "Timestamp" in df.columns:
        x_vals=df["Timestamp"].values
        x_label="Timestamp"
    else:
        x_vals=np.arange(len(df))
        x_label="Index"

    plt.figure(figsize=(10,5))
    if interest is not None:
        plt.plot(x_vals,interest,label="Interest(avg)",color='blue')
    if confusion is not None:
        plt.plot(x_vals,confusion,label="Confusion(avg)",color='red')
    chart_title = f"{sheet_name}: Interest & Confusion Over Time"
    plt.xlabel(x_label)
    plt.ylabel("Emotion Signal")
    plt.legend()
    finalize_plot_and_save(sheet_name, chart_title)

def do_correlation_matrices(df,sheet_name):
    facial=[c for c in df.columns if "(Facial)" in c]
    speech=[c for c in df.columns if "(Speech)" in c]
    lang=[c for c in df.columns if "(Language)" in c]

    #(Facial vs. Language)
    df_f_l=df[facial+lang].copy()
    for c in df_f_l.columns:
        df_f_l[c]=pd.to_numeric(df_f_l[c],errors='coerce')
    df_f_l.dropna(how='all',inplace=True)
    if df_f_l.shape[1]>=2:
        plt.figure(figsize=(8,6))
        corr=df_f_l.corr()
        sns.heatmap(corr,annot=True,cmap='coolwarm',fmt=".1f")
        chart_title=f"{sheet_name}: Corr (Facial vs Language)"
        finalize_plot_and_save(sheet_name, chart_title)

    #(Facial vs. Speech)
    df_f_s=df[facial+speech].copy()
    for c in df_f_s.columns:
        df_f_s[c]=pd.to_numeric(df_f_s[c],errors='coerce')
    df_f_s.dropna(how='all',inplace=True)
    if df_f_s.shape[1]>=2:
        plt.figure(figsize=(8,6))
        corr=df_f_s.corr()
        sns.heatmap(corr,annot=True,cmap='coolwarm',fmt=".1f")
        chart_title=f"{sheet_name}: Corr (Facial vs Speech)"
        finalize_plot_and_save(sheet_name, chart_title)

def dummy_confusion_matrix(df,sheet_name):
    if len(df)<6:
        return
    if "GroundTruthEmotion" not in df.columns or "PredictedEmotion" not in df.columns:
        import random
        emos=["Happy","Sad","Confused"]
        gl,pl=[],[]
        for _ in range(len(df)):
            gl.append(random.choice(emos))
            pl.append(random.choice(emos))
        df["GroundTruthEmotion"]=gl
        df["PredictedEmotion"]=pl

    from sklearn.metrics import confusion_matrix, classification_report
    valid=df.dropna(subset=["GroundTruthEmotion","PredictedEmotion"])
    if len(valid)==0:
        return
    ytrue=valid["GroundTruthEmotion"]
    ypred=valid["PredictedEmotion"]
    labs=sorted(list(set(ytrue)|set(ypred)))
    cmat=confusion_matrix(ytrue,ypred,labels=labs)

    print(f"\n=== {sheet_name} Dummy Confusion Matrix ===")
    print("Labels:", labs)
    import numpy as np
    print(np.around(cmat.astype(float),1))
    print("Report =>")
    print(classification_report(ytrue,ypred,labels=labs,digits=1))

def quick_lda_with_sentiment(df,sheet_name,n_topics=5):
    if "Thought Data" not in df.columns:
        return
    texts=df["Thought Data"].fillna("").astype(str).tolist()
    texts=[t.strip() for t in texts if t.strip()]
    if len(texts)<5:
        return

    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    vec=CountVectorizer(stop_words='english', max_features=500)
    X=vec.fit_transform(texts)
    if X.shape[0]<5:
        return
    lda=LatentDirichletAllocation(n_components=n_topics,random_state=999)
    lda.fit(X)
    feats=vec.get_feature_names_out()

    print(f"--- {sheet_name}: Quick LDA ---")
    for i,comp in enumerate(lda.components_):
        topidx=np.argsort(comp)[::-1][:10]
        topw=[feats[x] for x in topidx]
        print(f"Topic {i}: {topw}")

def volatility_bar_charts(df, sheet_name):
    """
    computes emotion 'volatility' using the difference (returns) approach:
      volatility = stdev_of_differences * sqrt(T)
    T = number of changes = len(ser) - 1
    Inspired by Watson & Naragon-Gainey (2014) emphasis on emotional variability.
    """
    emos = ["Interest","Boredom","Happiness","Anger",
            "Surprise","Disappointment","Satisfaction","Confusion"]

    vol_dict = {}

    for e in emos:
        ser = average_modalities(df, e)
        if ser is not None:
            arr = ser.dropna().values
            if len(arr) < 2:
                continue
            diffs = np.diff(arr)
            sigma = np.std(diffs, ddof=1)  
            T = len(diffs)

            sigma_T = sigma * np.sqrt(T)
            vol_dict[e] = sigma_T

    if not vol_dict:
        print(f"{sheet_name}: No valid emotion data for volatility.")
        return

    plt.figure(figsize=(10,5))
    sns.barplot(x=list(vol_dict.keys()), y=list(vol_dict.values()))
    chart_title = f"{sheet_name}: Emotion Volatility"
    plt.xlabel("Emotion")
    plt.ylabel("Volatility")
    plt.title(chart_title)
    plt.xticks(rotation=45)
    finalize_plot_and_save(sheet_name, chart_title)


##################
# Action Analysis
#########
def preprocess_multi_action(df):
    relevant=["Timestamp","Action Data"]
    df_sub=df.loc[:, [c for c in relevant if c in df.columns]].copy()
    df_sub=df_sub.loc[:, ~df_sub.columns.duplicated()]

    if "Action Data" not in df_sub.columns:
        return pd.DataFrame()

    df_sub['Action List']=df_sub['Action Data'].astype(str).str.split('|')
    df_exploded=df_sub.explode('Action List').reset_index(drop=True)
    df_exploded['Action List']=df_exploded['Action List'].str.strip()
    return df_exploded

def gather_actions_ordered_tds_sheets(data_dict):
    seen=set()
    ordered=[]
    for sh, df in data_dict.items():
        if "Action Data" not in df.columns:
            continue
        splitted=preprocess_multi_action(df)
        for act in splitted['Action List'].dropna():
            if act.lower()=="no action":
                continue
            if act not in seen:
                seen.add(act)
                ordered.append(act)
    return ordered

def generate_placeholders(unique_actions):
    mapping={}
    for act in unique_actions:
        if not act or act.lower()=="no action":
            continue
        a_lower=act.lower()
        if "typing init" in a_lower:
            code="TIx"
        elif "typing compl" in a_lower:
            code="TCx"
        elif "copy" in a_lower or "paste" in a_lower:
            code="CPx"
        elif "click" in a_lower:
            code="CLx"
        else:
            words=act.split("(")[0].strip().split()
            acronym="".join(w[0].upper() for w in words if w)+"x"
            code=acronym
        if (("(" in act or "'" in act)) and not code.startswith("CLx"):
            code+="*"
        mapping[act]=code
    if "No action" not in mapping:
        mapping["No action"]="No action"
    return mapping

def assign_codes_to_sheet(df, placeholder_map):
    splitted=preprocess_multi_action(df)
    if splitted.empty:
        return splitted
    splitted['Action Coding']=splitted['Action List'].map(placeholder_map).fillna("No action")
    splitted['Action Code']=splitted['Action Coding']
    return splitted

def finalize_sequential_codes(df_actions):
    if df_actions.empty:
        return df_actions
    counters={}
    new=[]
    for c in df_actions['Action Code']:
        if c=="No action":
            new.append(c)
            continue
        suffix=""
        if c.endswith("*"):
            suffix="*"
            c=c[:-1]
        if c.endswith("x"):
            base=c[:-1]
            counters.setdefault(base,0)
            counters[base]+=1
            newc=base+str(counters[base])+suffix
        else:
            counters.setdefault(c,0)
            counters[c]+=1
            newc=c+str(counters[c])+suffix
        new.append(newc)
    df_actions['Action Code']=new
    return df_actions

def horizontal_timeline_actions(df_actions, sheet_name):
    if df_actions.empty:
        return
    if "Timestamp" not in df_actions.columns:
        return
    codes=df_actions['Action Code'].unique()
    c2x={co:i for i,co in enumerate(codes)}
    xvals=[c2x[c] for c in df_actions['Action Code']]
    yvals=df_actions['Timestamp']

    plt.figure(figsize=(16,9))
    plt.scatter(xvals,yvals,s=40,alpha=0.8)
    chart_title=f"{sheet_name}: Horizontal Timeline (Seq Action Codes)"
    plt.xlabel("Action Code (Categorical Index)")
    plt.ylabel("Timestamp")
    ax=plt.gca()
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x',length=0)
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes,rotation=90,fontsize=7,fontweight='bold')
    labs=ax.get_xticklabels()
    for i,l in enumerate(labs):
        if i%2==0:
            l.set_y(-0.05)
        else:
            l.set_y(0.05)
    plt.subplots_adjust(bottom=0.3)
    finalize_plot_and_save(sheet_name, chart_title)

def freq_placeholder_codes(df_actions, sheet_name):
    if df_actions.empty:
        return
    if 'Action Coding' not in df_actions.columns:
        return
    freq=df_actions['Action Coding'].value_counts()
    plt.figure(figsize=(10,6))
    sns.barplot(x=freq.index,y=freq.values,palette='viridis')
    chart_title=f"{sheet_name}: Action Frequency (Action Coding)"
    plt.xlabel("Action Coding")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    finalize_plot_and_save(sheet_name, chart_title)

###########################################
# Additional color-coded emotion plots
###########
def plot_action_emotion_horizontal(df_actions, df_full, sheet_name, emotion="Interest"):
    if df_actions.empty or "Timestamp" not in df_actions.columns:
        return
    merged=pd.merge(df_actions, df_full, on="Timestamp", how="left")
    ser=average_modalities(merged,emotion)
    if ser is None:
        return
    merged[emotion+"_Value"]=ser.values
    codes=merged['Action Code'].unique()
    c2x={co:i for i,co in enumerate(codes)}
    merged['ActionX']=[c2x[c] for c in merged['Action Code']]

    plt.figure(figsize=(16,9))
    sc=plt.scatter(merged['ActionX'], merged['Timestamp'],
                   c=merged[emotion+"_Value"], cmap='coolwarm', s=40, alpha=0.8)
    plt.colorbar(sc,label=f"{emotion} (avg) Value")
    chart_title=f"{sheet_name}: Horizontal Timeline color by {emotion}"
    plt.xlabel("Action Code (index)")
    plt.ylabel("Timestamp")
    ax=plt.gca()
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x',length=0)
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes,rotation=90,fontsize=7,fontweight='bold')
    labs=ax.get_xticklabels()
    for i,lbl in enumerate(labs):
        if i%2==0:
            lbl.set_y(-0.05)
        else:
            lbl.set_y(0.05)
    plt.subplots_adjust(bottom=0.3)
    finalize_plot_and_save(sheet_name, chart_title)

def plot_action_emotion_horizontal_posneg(df_actions, df_full, sheet_name):
    if df_actions.empty or "Timestamp" not in df_actions.columns:
        return
    merged=pd.merge(df_actions, df_full, on="Timestamp", how="left")

    pos_set=["Happiness","Satisfaction","Interest"]
    neg_set=["Anger","Disappointment", "Boredom"]

    def avg_set(df_,emoset):
        valid=[]
        for emo in emoset:
            s=average_modalities(df_,emo)
            if s is not None:
                valid.append(s)
        if not valid:
            return pd.Series([0]*len(df_),index=df_.index)
        cat=pd.concat(valid,axis=1)
        return cat.mean(axis=1,skipna=True)

    pos_series=avg_set(merged,pos_set)
    neg_series=avg_set(merged,neg_set)
    merged["PosNegValue"]=(pos_series - neg_series).values

    codes=merged['Action Code'].unique()
    c2x={co:i for i,co in enumerate(codes)}
    merged["ActionX"]=[c2x[c] for c in merged['Action Code']]

    plt.figure(figsize=(16,9))
    sc=plt.scatter(merged['ActionX'], merged['Timestamp'],
                   c=merged["PosNegValue"], cmap='coolwarm', s=150, alpha=0.9) # f
    plt.colorbar(sc,label="(Positivity - Negativity) Emotions")
    chart_title=f"{sheet_name}: Horizontal Timeline color by (Positivity - Negativity)"
    plt.xlabel("Action Code (index)")
    plt.ylabel("Timestamp")
    ax=plt.gca()
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x',length=0)
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes,rotation=90,fontsize=7,fontweight='bold')
    labs=ax.get_xticklabels()
    for i,lbl in enumerate(labs):
        if i%2==0:
            lbl.set_y(-0.05)
        else:
            lbl.set_y(0.05)
    plt.subplots_adjust(bottom=0.3)
    finalize_plot_and_save(sheet_name, chart_title)

###########################################
#Academic vs NonAcademic
###############
def plot_academic_vs_nonacademic_actions(data_dict):
    acad_frames=[]
    non_frames=[]
    for sh,df in data_dict.items():
        if sh.startswith("A"):
            acad_frames.append(df)
        elif sh.startswith("NA"):
            non_frames.append(df)

    if not acad_frames and not non_frames:
        return

    all_actions=[]
    for sh,df in data_dict.items():
        splitted=preprocess_multi_action(df)
        for act in splitted['Action List'].dropna():
            if act.lower()=="no action":continue
            all_actions.append(act)
    all_unique=list(dict.fromkeys(all_actions))
    placeholder_map=generate_placeholders(all_unique)

    acad_actions=None
    non_actions=None

    if acad_frames:
        cfr=[]
        for df in acad_frames:
            splitted=assign_codes_to_sheet(df,placeholder_map)
            splitted=finalize_sequential_codes(splitted)
            splitted["Category"]="Academic"
            cfr.append(splitted)
        acad_actions=pd.concat(cfr,ignore_index=True)
    if non_frames:
        cfr=[]
        for df in non_frames:
            splitted=assign_codes_to_sheet(df,placeholder_map)
            splitted=finalize_sequential_codes(splitted)
            splitted["Category"]="NonAcademic"
            cfr.append(splitted)
        non_actions=pd.concat(cfr,ignore_index=True)

    if acad_actions is None and non_actions is None:
        return

    fig,axs=plt.subplots(1,2,figsize=(16,9),sharey=True)
    if acad_actions is not None and not acad_actions.empty:
        codes=acad_actions['Action Code'].unique()
        c2x={co:i for i,co in enumerate(codes)}
        xvals=[c2x[c] for c in acad_actions['Action Code']]
        yvals=acad_actions['Timestamp']
        axs[0].scatter(xvals,yvals,s=40,alpha=0.8,c='blue')
        axs[0].set_title("Academic Actions (All 'A...') Sheets", pad=5)
        axs[0].set_xlabel("Action Code (index)")
        axs[0].spines['bottom'].set_visible(False)
        axs[0].tick_params(axis='x',length=0)
        axs[0].set_xticks(range(len(codes)))
        axs[0].set_xticklabels(codes,rotation=90,fontsize=7,fontweight='bold')
        labs=axs[0].get_xticklabels()
        for i,l in enumerate(labs):
            if i%2==0:
                l.set_y(-0.05)
            else:
                l.set_y(0.05)

    if non_actions is not None and not non_actions.empty:
        codes2=non_actions['Action Code'].unique()
        c2x2={co:i for i,co in enumerate(codes2)}
        xvals2=[c2x2[c] for c in non_actions['Action Code']]
        yvals2=non_actions['Timestamp']
        axs[1].scatter(xvals2,yvals2,s=40,alpha=0.8,c='red')
        axs[1].set_title("Non-Academic Actions (All 'NA...') Sheets", pad=5)
        axs[1].set_xlabel("Action Code (index)")
        axs[1].spines['bottom'].set_visible(False)
        axs[1].tick_params(axis='x',length=0)
        axs[1].set_xticks(range(len(codes2)))
        axs[1].set_xticklabels(codes2,rotation=90,fontsize=7,fontweight='bold')
        labs2=axs[1].get_xticklabels()
        for i2,lbl2 in enumerate(labs2):
            if i2%2==0:
                lbl2.set_y(-0.05)
            else:
                lbl2.set_y(0.05)

    axs[0].set_ylabel("Timestamp")
    plt.subplots_adjust(bottom=0.3)
    chart_title="Academic_vs_NonAcademic_Horizontal_Timeline"
    finalize_plot_and_save("Academic_vs_NonAcademic", chart_title)


#Display Action Coding Table PER-SHEET
###########################################
def display_placeholder_code_table_modified_matplot(placeholder_map, sheet_name):
    """
    - TIx => single row => "Typing Initiation of prompt x"
    - TCx => single row => "Typing Completion of prompt x"
    - CPx => repeat all distinct actions that map to CPx
    - CLx => same repeating logic as CPx
    - Others => just one row => first action + '(sample)'
    - If "nan (sample)" => "No Action (sample)"
    - Title => "Action Code Table for <sheet_name>" 

    """
    from collections import defaultdict


    code_to_actions = defaultdict(list)
    for orig_action, code in placeholder_map.items():
        if not code or code.lower()=="no action" or code.lower()=="nan":
            continue
        code_to_actions[code].append(orig_action)

    
    table_rows=[]
    sorted_codes=sorted(code_to_actions.keys())
    for code in sorted_codes:
        if code=="TIx":
            desc="Typing Initiation of prompt x"
            table_rows.append([desc, code])
            continue
        if code=="TCx":
            desc="Typing Completion of prompt x"
            table_rows.append([desc, code])
            continue

        if code in ("CPx","CLx"):
            
            for oa in code_to_actions[code]:
                row_desc = oa+" (sample)"
                if row_desc.lower().startswith("nan"):
                    row_desc="No Action (sample)"
                table_rows.append([row_desc, code])
            continue

       
        arr=code_to_actions[code]
        if not arr:
            continue
        row_desc = arr[0] + " (sample)"
        if row_desc.lower().startswith("nan"):
            row_desc="No Action (sample)"
        table_rows.append([row_desc, code])

    df_out = pd.DataFrame(table_rows, columns=["Action Description","Action Coding"])

    fig, ax = plt.subplots(figsize=(12, max(3,len(df_out)*0.4+1)))
    ax.axis('tight')
    ax.axis('off')

    the_table = ax.table(
        cellText=df_out.values,
        colLabels=df_out.columns,
        loc='center'
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)

    chart_title = f"Action Code Table for {sheet_name}"
    ax.set_title(chart_title, pad=-2)  

    plt.tight_layout()

    
    safe_title = re.sub(r'[\\/*?:"<>|]', '_', chart_title)
    safe_title = safe_title.replace(' ', '_')
    filename = f"{safe_title}.png"
    plt.savefig(filename,dpi=150)
    plt.show()

   
    global FIGURE_LOG
    FIGURE_LOG.append((sheet_name, filename))

############
#MAIN
###########################################
def main():
    excel_file="Human AI Record Template_V16.xlsx"
    sheets_of_focus=[
        "A1 - Cellular Automata Model",
        "A2 - MP Birthplace Analysis",
        "A3 - Deep Learning Model",
        "NA1 - Internship Preparation",
        "NA2 - Christmas Gift Selection",
        "NA3 - Event Coordination"
    ]
    data_dict=read_and_unify_sheets_tds(excel_file,sheets_of_focus)

 
    for sheet_name, df in data_dict.items():
    
        plot_interest_boredom(df,sheet_name)
        plot_interest_confusion(df,sheet_name)
        do_correlation_matrices(df,sheet_name)
        dummy_confusion_matrix(df,sheet_name)
        quick_lda_with_sentiment(df,sheet_name,n_topics=5)
        volatility_bar_charts(df,sheet_name)

   
    for sheet_name, df in data_dict.items():
        splitted=preprocess_multi_action(df)
        actions_sheet=splitted["Action List"].dropna().unique()
        if len(actions_sheet)==0:
            print(f"{sheet_name}: No actions => skipping.")
            continue

    
        placeholders_sheet={}
        for a in actions_sheet:
            a_lower=a.lower()
            if "typing init" in a_lower:
                code="TIx"
            elif "typing compl" in a_lower:
                code="TCx"
            elif "copy" in a_lower or "paste" in a_lower:
                code="CPx"
            elif "click" in a_lower:
                code="CLx"
            else:
                words=a.split("(")[0].strip().split()
                acronym="".join(w[0].upper() for w in words if w)+"x"
                code=acronym
            if (("(" in a or "'" in a)) and not code.startswith("CLx"):
                code+="*"
            placeholders_sheet[a]=code

   
        display_placeholder_code_table_modified_matplot(
            placeholders_sheet, sheet_name=sheet_name
        )

      
        df_actions=assign_codes_to_sheet(df, placeholders_sheet)
        if df_actions.empty:
            continue
        df_actions=finalize_sequential_codes(df_actions)

        horizontal_timeline_actions(df_actions,sheet_name)
        freq_placeholder_codes(df_actions,sheet_name)
        plot_action_emotion_horizontal(df_actions, df, sheet_name, emotion="Interest")
        plot_action_emotion_horizontal(df_actions, df, sheet_name, emotion="Boredom")
        plot_action_emotion_horizontal_posneg(df_actions, df, sheet_name)

    
    plot_academic_vs_nonacademic_actions(data_dict)

    
    export_figure_list_to_doc(FIGURE_LOG, "Plot_Figures.doc")


if __name__=="__main__":
    main()
