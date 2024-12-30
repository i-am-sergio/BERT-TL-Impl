#@title Import python packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

from fastai import *
from fastai.text import *
from fastai.callbacks import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

from pytorch_transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig
from pytorch_transformers import AdamW

from fastprogress import master_bar, progress_bar
from datetime import datetime



class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

class Fold(Enum):
  No = 1
  TenFold = 2
  ProjFold = 3

class Sampling(Enum):
  NoSampling = 1
  UnderSampling = 2
  OverSampling = 3

config = Config(
    num_labels = 2, # will be set automatically afterwards
    model_name="bert-base-cased", # bert_base_uncased, bert_large_cased, bert_large_uncased
    max_lr=2e-5, # default: 2e-5
    moms=(0.8, 0.7), # default: (0.8, 0.7); alt.(0.95, 0.85)
    epochs=16, # 10, 16, 32, 50
    bs=16, # default: 16
    weight_decay = 0.01,
    max_seq_len=128, # 50, 128
    train_size=0.75, # 0.8
    loss_func=nn.CrossEntropyLoss(),
    seed=904727489, #default: 904727489, 42 (as in Dalpiaz) or None
    es = False, # True
    min_delta = 0.01,
    patience = 3,
    fold = Fold.No, # Fold.No, Fold.TenFold, Fold.ProjFold
    sampling = Sampling.NoSampling, #Sampling.UnderSampling, Sampling.NoSampling, Sampling.OverSampling
)

clazz = 'NFR' # class to train classification on

config_data = Config(
    root_folder = '.', # where is the root folder? Keep it that way if you want to load from Google Drive
    data_folder = '/', # where is the folder containing the datasets; relative to root
    train_data = ['promise_nfr.csv'], # dataset file to use
    label_column = clazz,
    log_folder_name = '/log/',
    log_file = clazz + '_' + Fold(config.fold).name + '_' + Sampling(config.sampling).name + '_classifierPredictions_' + datetime.now().strftime('%Y%m%d-%H%M') + '.txt', # log-file name (make sure log folder exists)
    result_file = clazz + '_' + Fold(config.fold).name + '_' + Sampling(config.sampling).name + '_classifierResults_' + datetime.now().strftime('%Y%m%d-%H%M') + '.txt', # result-file name (make sure log folder exists)
    model_path = '/models/', # where is the folder for the model(s); relative to the root
    model_name = 'NoRBERT.pkl', # what is the model name?
    gdrive_root_folder = '/content/drive/My Drive/Code/Task1_to_3_original_Promise_NFR_dataset/', # Set this to the Google Drive path. Starts with '/content/drive/' and then usually 'My Drive/*' for the files in your Drive

    orig_data_set_zip = 'https://zenodo.org/record/8347866/files/NoRBERT_RE20_Paper65.zip', # link to the data set (on zenodo). DO NOT CHANGE!
    orig_data_zip_name = 'NoRBERT_RE20_Paper65.zip', # DO NOT CHANGE
    orig_data_file_in_zip = 'Code/Task1_to_3_original_Promise_NFR_dataset/promise_nfr.csv', # DO NOT CHANGE

    # Project split to use, either p-fold (as in Dalpiaz) or loPo
    #project_fold = [[3, 9, 11], [1, 5, 12], [6, 10, 13], [1, 8, 14], [3, 12, 15], [2, 5, 11], [6, 9, 14], [7, 8, 13], [2, 4, 15], [4, 7, 10] ], # p-fold
    project_fold = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15] ], # loPo
)

load_from_gdrive = False # True, if you want to use Google Drive; else, False
save_model = False # True, if you want to use save the model file (make sure the "models" folder exists)


#@title Prepare data loading: Init loading from Google Drive, if set in config above. Else, download the data set from zenodo (using wget) {display-mode: "form"}
if load_from_gdrive:
    from google.colab import drive
    # Connect to drive to load the corpus from there
    drive.mount('/content/drive', force_remount=True)
    config_data.root_folder = config_data.gdrive_root_folder
else:
    # If the file does not exist already, download the zip and extract the needed file
    data_path = config_data.root_folder + config_data.data_folder + config_data.train_data[0]
    data_file = Path(data_path)
    if not data_file.exists():
        !wget {config_data.orig_data_set_zip}
        import zipfile
        with zipfile.ZipFile(config_data.orig_data_zip_name) as z:
            with open(data_path, 'wb') as f:
                f.write(z.read(config_data.orig_data_file_in_zip))


#@title Define logging functions and seed generation {display-mode: "form"}
def initLog():
    logfolder = config_data.root_folder + config_data.log_folder_name

    if not os.path.isdir(logfolder):
      print("Log folder does not exist, trying to create folder.")
      try:
        os.mkdir(logfolder)
      except OSError:
        print ("Creation of the directory %s failed" % logfolder)
      else:
        print ("Successfully created the directory %s" % logfolder)
    logfile = logfolder + config_data.log_file
    log_txt = datetime.now().strftime('%Y-%m-%d %H:%M') + ' ' + get_info()
    with open(logfile, 'w') as log:
        log.write(log_txt + '\n')

def logLine(line):
    logfile = config_data.root_folder + config_data.log_folder_name  + config_data.log_file
    with open(logfile, 'a') as log:
        log.write(line + '\n')

def logResult(result):
    logfile = config_data.root_folder + config_data.log_folder_name + config_data.result_file
    with open(logfile, 'a') as log:
        log.write(get_info() + '\n')
        log.write(result + '\n')

def get_info():
     model_config = 'model: {}, max_lr: {}, epochs: {}, bs: {}, train_size: {}, weight decay: {},  Seed: {}, Data: {}, Column: {}, EarlyStopping: {}:{};pat:{}'.format(config.model_name, config.max_lr, config.epochs, config.bs, config.train_size, config.weight_decay, config.seed, config_data.train_data, config_data.label_column, config.es, config.min_delta, config.patience)
     return model_config

def set_seed(seed):
    if seed is None:
        seed = random.randint(0, 2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

set_seed(config.seed)


#@title Create proper tokenizer for our data (adapting FastAiTokenizer to use BertTokenizer) {display-mode: "form"}
class FastAiBertTokenizer(BaseTokenizer):
    """Wrapper around BertTokenizer to be compatible with fast.ai"""
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=512, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t:str):
        """Limits the maximum sequence length. Prepend with [CLS] and append [SEP]"""
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]




#@title Define Processors and Databunch {display-mode: "form"}
class BertTokenizeProcessor(TokenizeProcessor):
    """Special Tokenizer, where we remove sos/eos tokens since we add that ourselves in the tokenizer."""
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)

class BertNumericalizeProcessor(NumericalizeProcessor):
    """Use a custom vocabulary to match the original BERT model."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, vocab=Vocab(list(bert_tok.vocab.keys())), **kwargs)

def get_bert_processor(tokenizer:Tokenizer=None, vocab:Vocab=None):
    return [BertTokenizeProcessor(tokenizer=tokenizer),
            NumericalizeProcessor(vocab=vocab)]

class BertDataBunch(TextDataBunch):
    @classmethod
    def from_df(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
              tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
              label_cols:IntsOrStrs=0, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from DataFrames."
        p_kwargs, kwargs = split_kwargs_by_func(kwargs, get_bert_processor)
        # use our custom processors while taking tokenizer and vocab as kwargs
        processor = get_bert_processor(tokenizer=tokenizer, vocab=vocab, **p_kwargs)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                      TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_df(cols=label_cols, classes=classes)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        return src.databunch(**kwargs)
    

#@title Define own BertTextClassifier class{display-mode: "form"}
class BertTextClassifier(BertPreTrainedModel):
    def __init__(self, model_name, num_labels):
        config = BertConfig.from_pretrained(model_name)
        super(BertTextClassifier, self).__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(model_name, config=config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)


    def forward(self, tokens, labels=None, position_ids=None, token_type_ids=None, attention_mask=None, head_mask=None):
        outputs = self.bert(tokens, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=head_mask)

        pooled_output = outputs[1]

        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)

        activation = nn.Softmax(dim=1)
        probs = activation(logits)

        return logits
    

#@title Define functions to load data {display-mode: "form"}
def load_data(filename):
    fpath = config_data.root_folder + config_data.data_folder + filename
    print(fpath)
    df = pd.read_csv(fpath, delimiter=';', header=0, encoding='utf8', names=['number', 'ProjectID', 'RequirementText', 'class', 'NFR', 'F', 'A', 'FT', 'L', 'LF', 'MN', 'O', 'PE', 'PO', 'SC', 'SE', 'US'])
    df = df.dropna()
    return df

def load_all_data(filenames):
    df = load_data(filenames[0])
    for i in range(1, len(filenames)):
        df = df.append(load_data(filenames[i]))
    return df



#@title Actually load the dataset{display-mode: "form"}
# load the train dataset
df = load_all_data(config_data.train_data)
input_col = 'RequirementText'
# shuffle the dataset a bit and get the amount of classes
df = df.sample(frac=1, axis=0, random_state = config.seed)
config.num_labels = df[config_data.label_column].nunique()

print(df.shape)
print(df[config_data.label_column].value_counts())


#@title Create the dictionary that contains the labels along with their indices. This is useful for evaluation and similar. {display-mode: "form"}
def create_label_indices(df):
    #prepare label
    labels = ['not_' + config_data.label_column, config_data.label_column]

    #create dict
    labelDict = dict()
    for i in range (0, len(labels)):
        labelDict[i] = labels[i]
    return labelDict

label_indices = create_label_indices(df)
print(label_indices)


#@title Define functions for under-/oversample dataset {display-mode: "form"}
def undersample(df_trn, major_label, minor_label):
  sample_size = sum(df_trn[config_data.label_column] == minor_label)
  majority_indices = df_trn[df_trn[config_data.label_column] == major_label].index
  random_indices = np.random.choice(majority_indices, sample_size, replace=False)
  sample = df_trn.loc[random_indices]
  sample = sample.append(df_trn[df_trn[config_data.label_column] == minor_label])
  df_trn = sample
  df_trn = df_trn.sample(frac=1, axis=0, random_state = config.seed)
  print(df_trn[config_data.label_column].value_counts())
  return df_trn

def oversample(df_trn, major_label, minor_label):
  minor_size = sum(df_trn[config_data.label_column] == minor_label)
  major_size = sum(df_trn[config_data.label_column] == major_label)
  multiplier = major_size//minor_size
  sample = df_trn
  minority_indices = df_trn[df_trn[config_data.label_column] == minor_label].index
  diff = major_size - (multiplier * minor_size)
  random_indices = np.random.choice(minority_indices, diff, replace=False)
  sample = pd.concat([df_trn.loc[random_indices], sample], ignore_index=True)
  for i in range(multiplier - 1):
    sample = pd.concat([sample, df_trn[df_trn[config_data.label_column] == minor_label]], ignore_index=True)
  df_trn = sample
  df_trn = df_trn.sample(frac=1, axis=0, random_state = config.seed)
  print(df_trn[config_data.label_column].value_counts())
  return df_trn

#@title Function to split dataframe according to Sampling strategy and train size {display-mode: "form"}
def split_dataframe(df, train_size = 0.8, random_state = None):
    # split data into training and validation set
    df_trn, df_valid = train_test_split(df, stratify = df[config_data.label_column], train_size = train_size, random_state = random_state)
    # apply sample strategy
    sizeOne = sum(df_trn[config_data.label_column] == 1)
    sizeZero = sum(df_trn[config_data.label_column] == 0)
    major_label = 0
    minor_label = 1
    if sizeOne > sizeZero:
      major_label = 1
      minor_label = 0
    if config.sampling == Sampling.UnderSampling:
      df_trn = undersample(df_trn, major_label, minor_label)
    elif config.sampling == Sampling.OverSampling:
      df_trn = oversample(df_trn, major_label, minor_label)
    return df_trn, df_valid


#@title Create a predictor class{display-mode: "form"}
class Predictor:
    def __init__(self, classifier):
        self.classifier = classifier
        self.classes = self.classifier.data.classes

    def predict(self, text):
        prediction = self.classifier.predict(text)

    
#@title Define functions to create databunch, learner and actual classifier{display-mode: "form"}
def create_databunch(config, df_trn, df_valid):
    bert_tok = BertTokenizer.from_pretrained(config.model_name,)
    fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config.max_seq_len), pre_rules=[], post_rules=[])
    fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
    return BertDataBunch.from_df(".",
                   train_df=df_trn,
                   valid_df=df_valid,
                   tokenizer=fastai_tokenizer,
                   vocab=fastai_bert_vocab,
                   bs=config.bs,
                   text_cols=input_col,
                   label_cols=config_data.label_column,
                   collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
              )


def create_learner(config, databunch):
    model = BertTextClassifier(config.model_name, config.num_labels)

    optimizer = partial(AdamW)
    if config.es:
      learner = Learner(
        databunch, model,
        optimizer,
        wd = config.weight_decay,
        metrics=FBeta(beta=1), #accuracy, (metric to optimize on)
        loss_func=config.loss_func, callback_fns=[partial(EarlyStoppingCallback, monitor='f_beta', min_delta=config.min_delta, patience=config.patience)]
      )
    else:
      learner = Learner(
        databunch, model,
        optimizer,
        wd = config.weight_decay,
        metrics=FBeta(beta=1), #accuracy, (metric to optimize on)
        loss_func=config.loss_func,
      )

    return learner

# Create the classifier
def create_classifier(config, df):
  df_trn, df_valid = split_dataframe(df, train_size = config.train_size, random_state = config.seed)
  databunch = create_databunch(config, df_trn, df_valid)

  return create_learner(config, databunch)



#@title Define predict loop {display-mode: "form"}
def predict_and_log_result(classifier, df_eval):
  predictor = Predictor(classifier)
  flat_predictions, flat_true_labels = [], []
  column_index = df_eval.columns.get_loc(config_data.label_column)

  for row in progress_bar(df_eval.itertuples(), total=len(df_eval)):
      class_text = row.RequirementText
      class_label = row[column_index+1]
      flat_true_labels.append(class_label)
      prediction = predictor.predict(class_text)
      flat_predictions.append(prediction)

      log_text = 'PID: {}, {}, {} -> {}'.format(row.ProjectID, class_text, label_indices.get(class_label), label_indices.get(prediction))
      logLine(log_text)

  # get labels in correct order
  target_names = []
  test_labels = unique_labels(flat_true_labels, flat_predictions)
  test_labels = np.sort(test_labels)
  for x in test_labels:
    target_names.append(label_indices.get(x))

  result = classification_report(flat_true_labels, flat_predictions, target_names=target_names, digits = 5)
  logResult(result)
  print(result)
  return flat_predictions, flat_true_labels



#@title Define train and test loop{display-mode: "form"}
def train_and_predict(df_train, df_eval, overall_flat_predictions, overall_flat_true_labels, results):
  classifier = create_classifier(config, df_train)
  # Train the classifier on train set
  print("Training classifier...")
  print(classifier.fit_one_cycle(config.epochs, max_lr=config.max_lr, moms=config.moms, wd=config.weight_decay))
  
  #Predict on test set
  flat_predictions, flat_true_labels = predict_and_log_result(classifier, df_eval)

  # Extender listas globales
  overall_flat_predictions.extend(flat_predictions)
  overall_flat_true_labels.extend(flat_true_labels)
  
  # Imprimir formas y tipos de datos para depuración
  print("Forma de flat_true_labels:", np.shape(flat_true_labels))
  print("Forma de flat_predictions:", np.shape(flat_predictions))
  print("Tipo de flat_true_labels:", type(flat_true_labels))
  print("Tipo de flat_predictions:", type(flat_predictions))

  test_labels = df_eval[config_data.label_column].unique()
  test_labels = np.sort(test_labels)

  # Asegúrate de que los resultados sean válidos
  results.extend(precision_recall_fscore_support(flat_true_labels, flat_predictions, labels = test_labels))
  return classifier, overall_flat_predictions, overall_flat_true_labels, results



#@title Decide how to fold and train the classifier {display-mode: "form"}
overall_flat_predictions, overall_flat_true_labels, results = [], [], []
initLog()

if config.fold == Fold.TenFold: # Use Stratified ten fold cross validation
  skf = StratifiedKFold(n_splits=10)
  fold_number = 1
  for train, test in skf.split(df, df[config_data.label_column]):
    df_train = df.iloc[train]
    df_eval = df.iloc[test]
    log_text = '/////////////////////// Fold: {} of {} /////////////////////////////'.format(fold_number,10)
    logLine(log_text)
    classifier, overall_flat_predictions, overall_flat_true_labels, results = train_and_predict(df_train, df_eval, overall_flat_predictions, overall_flat_true_labels, results)
    fold_number = fold_number + 1
elif config.fold == Fold.ProjFold: # Use project specific fold as described in config_data
  for k in config_data.project_fold:
    test = df.loc[df['ProjectID'].isin(k)].index
    train = df.loc[~df['ProjectID'].isin(k)].index
    df_train = df.loc[train]
    df_eval = df.loc[test]
    log_text = '/////////////////////// Test-Projects: {} /////////////////////////////'.format(k)
    logLine(log_text)
    classifier, overall_flat_predictions, overall_flat_true_labels, results = train_and_predict(df_train, df_eval, overall_flat_predictions, overall_flat_true_labels, results)
else: # Use train/test split
  df_train, df_eval = train_test_split(df,stratify=df[config_data.label_column], train_size=config.train_size, random_state= config.seed)  
  classifier, overall_flat_predictions, overall_flat_true_labels, results = train_and_predict(df_train, df_eval, overall_flat_predictions, overall_flat_true_labels, results)

get_memory_usage_str()