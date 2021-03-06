import json
import os
from model import SentenceVAE
from utils import to_var, idx2word, interpolate
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder




def generate(date, epoch, attribute_list, n_samples, dataset="imdb"): # TODO : uncomment - only for testing single attribute
#def generate(date, epoch, sentiment, n_samples):

    '''

    :param date: (str-filepath) which experiment we want to test
    :param epoch: (int) the epoch with best validation score obtained during the training
    :param attribute_list: [list] hot-one encodings of the attribute desired
    :param n_samples: [int] number of sentences wanted
    :param dataset: [str] 'yelp' - 'imdb'
    :return:
    '''
    date = date
    cuda2 = torch.device('cuda:0')
    epoch = epoch
    #date = "2020-Feb-26-17:47:47"
    #exp_descr = pd.read_csv("EXP_DESCR/" + date + ".csv")
    #print("Pretained: ", exp_descr['pretrained'][0])
    #print("Bidirectional: ", exp_descr['Bidirectional'][0])
    #epoch = str(10)
    #data_dir = 'data'
    #


    params = pd.read_csv("Parameters/params.csv")
    params = params.set_index('time')
    exp_descr = params.loc[date]
    # 2019-Dec-02-09:35:25, 60,300,256,0.3,0.5,16,False,0.001,10,False

    embedding_size = exp_descr["embedding_size"]
    hidden_size = exp_descr["hidden_size"]
    rnn_type = exp_descr['rnn_type']
    word_dropout = exp_descr["word_dropout"]
    embedding_dropout = exp_descr["embedding_dropout"]
    latent_size = exp_descr["latent_size"]
    num_layers = 1
    batch_size = exp_descr["batch_size"]
    bidirectional = bool(exp_descr["bidirectional"])
    max_sequence_length = exp_descr["max_sequence_length"]
    back = exp_descr["back"]
    attribute_size = exp_descr["attr_size"]
    wd_type = exp_descr["word_drop_type"]
    num_samples = 2 # TODO : is it to match the attr_size ?
    save_model_path = 'bin'
    ptb = False
    """
    if ptb == True:
        vocab_dir = '/ptb.vocab.json'
    else:
        vocab_dir = '/'+dataset+'_vocab.json'

    with open("bin/" + date+"/"+ vocab_dir, 'r') as file:
        vocab = json.load(file)
    """
    # TODO : From single attribute

    if ptb == True:
        vocab_dir = '/ptb.vocab.json'
    else:
        vocab_dir = '/yelp_vocab.json'

    # with open("bin/" + date+"/"+ vocab_dir, 'r') as file:
    #     vocab = json.load(file)

    with open("data/"+"/"+ vocab_dir, 'r') as file:
        print("file : ", file)
        vocab = json.load(file)


    w2i, i2w = vocab['w2i'], vocab['i2w']


    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=max_sequence_length,
        embedding_size=embedding_size,
        rnn_type=rnn_type,
        hidden_size=hidden_size,
        word_dropout=0,
        embedding_dropout=0,
        latent_size=latent_size,
        num_layers=num_layers,
        cuda = cuda2,
        bidirectional=bidirectional,
        attribute_size=5, # TODO : To change to 7 , 2 only for testing
        word_dropout_type='static',
        back=back
    )

    print(model)
    # Results
    # 2019-Nov-28-13:23:06/E4-5".pytorch"

    load_checkpoint = "bin/" + date + "/" + "E" + str(epoch) + ".pytorch"
    # load_checkpoint = "bin/2019-Nov-28-12:03:44 /E0.pytorch"

    if not os.path.exists(load_checkpoint):
        raise FileNotFoundError(load_checkpoint)

    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     device = "cuda"
    # else:
    #     device = "cpu"
    device = "cpu"

    model.load_state_dict(torch.load(load_checkpoint, map_location=torch.device(device)))
    def attr_generation(n):
        labels = np.random.randint(2, size=n)
        # print("labels ", labels)
        enc = OneHotEncoder(handle_unknown='ignore')
        labels = np.reshape(labels, (len(labels), 1))
        enc.fit(labels)
        one_hot = enc.transform(labels).toarray()
        one_hot = one_hot.astype(np.float32)
        one_hot = torch.from_numpy(one_hot)
        return one_hot

    model.eval()
    labels = attr_generation(n=num_samples)


    print('----------SAMPLES----------')
    labels = []
    generated = []
    for i in range(n_samples):
        samples, z, l = model.inference(attribute_list)
        # samples, z, l = model.inference_sent(sentiment)
        s = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])
        #print(sentiment_analyzer_scores(s[0]))
        #if sentiment_analyzer_scores(s[0])[1] == sentiment:
        generated.append(s[0])
        # print(s[0])
        # print(s)

        #labels.append(sentiment_analyzer_scores(s[0])[0])
        #print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
    print(sum(labels))
    translation = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])
    return generated



#date = "2020-Mar-26-16:25:48"
date = "2022-Apr-24-20:03:16" #"2022-Apr-24-17:54:51"#2022-Apr-23-04:17:35"#"2022-Apr-21-18:22:07" #"2022-Apr-21-16:26:47" #"2022-Apr-21-12:56:12" #imdb
#date = "2020-Mar-17-15:51:11"
#bin/2020-May-09-06:35:11
#date = "2020-May-09-06:35:11"
#date = "2020-May-02-11:06:34"
#date = "2020-May-10-10:30:35"
#date = "2020-May-10-14:14:47"



epoch = 29

# TODO: only for testing single attribute

samples = 100
#yelp
label = ["possingular", "negsingular", "posneutral",
"negneutral","posplural","negplural"]

# label = ["PresPosPlural", "PresPosSingular", "PresPosNeutral",
# "PastPosPlural","PastPosSingular","PastPosNeutral", "PresNegPlural",
# "PresNegSingular", "PresNegNeutral", "PastNegPlural", "PastNegSingular",  "PastNegNeutral"]


#imdb
#label = ["pastpossingular", "pastposplural", "prespossingular", "presposplural",
 #        "pastnegsingular", "pastnegplural", "presnegsingular", "presnegplural"]


def label_OneHotEncode(label):

    if label == "possingular":
        return [1, 0, 1, 0, 0]

    if label == "negsingular":
        return [0, 1, 1, 0, 0]

    if label == "posneutral":
        return [1, 0, 0, 1, 0]

    if label == "negneutral":
        return [0, 1, 0, 1, 0]

    if label == "posplural":
        return [1, 0, 0, 0, 1]

    if label == "negplural":
        return [0, 1, 0, 0, 1]

generated_sentences = []
y_generated = []
generated_sentences_all=[]
labels_true=[]
for l in label:
    print("label: ",l)
    g_sent = generate(date, epoch, l, samples, dataset="imdb")
    y_gen = [l]*len(g_sent)
    generated_sentences.append(g_sent)
    y_generated.append(y_gen)
    print("sentence: ", g_sent)

    # label OneHotEncoding
    # labels_samples = label_OneHotEncode(l)
    for i in range(samples):
        labels_true.append(label_OneHotEncode((l)))

labels_true = np.asarray(labels_true)
labels_true = pd.DataFrame({'Positive': labels_true[:,0],
                        'Negative': labels_true[:,1],
                       'Singular': labels_true[:,2],
                       'Neutral': labels_true[:,3],
                       'Plural': labels_true[:,4]})

print("generated_sentences", type(generated_sentences))
generated_sentences_all = [str(k).replace('<eos>', '').strip() for i in generated_sentences for k in i]
print("generated_sentences_all : ", generated_sentences_all)
generated_sentences = pd.DataFrame({'text': generated_sentences_all})

labels_true.to_csv("data/y_yelp_generated.csv", index=False)
generated_sentences.to_csv("data/yelp_generated.csv", index=False)

# for i in g_sent:
#     print("g_sent \n", i)


# generated_sent=generate(date, epoch,"Negative", 10)
# generated_sent=generate(date, epoch,"Positive", 10)
# generated_sent = generate(date, epoch,"Negative", 10)
# for i in generated_sentences:
#     print('generated_sentences : \n', i)