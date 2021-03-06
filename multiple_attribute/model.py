import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var
import time
import json
from Embedder import Embedder
import numpy as np
import torch.nn.functional as F # TODO : loss_rec
from torch.nn.functional import normalize



class SentenceVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                attribute_size, sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, cuda, word_dropout_type,back, num_layers=1, bidirectional=False, pretrained=False, embedding_layer=None):

        '''

        Sentence Variatiatonal Autoencoder Initialization

        :param vocab_size: vocabulary size
        :param embedding_size: the embedding size (default is 300)
        :param rnn_type: 'rnn, gru, lstm'
        :param hidden_size:
        :param word_dropout: 'only if cyclical not active'
        :param embedding_dropout: 'dropout to apply to the embeddings'
        :param latent_size: 'context-size dimension'
        :param attribute_size: 'number of attributes controlled'
        :param sos_idx: 'Start of Sentence index'
        :param eos_idx: 'End of Sentence index'
        :param pad_idx: 'Padding index'
        :param unk_idx: 'Unknown token index'
        :param max_sequence_length: 'Maximum processed Language'
        :param cuda: 'which GPU you want to use it'
        :param word_dropout_type: 'Standard or Cyclical'
        :param back: 'Context-Aware loss or not'
        :param num_layers:
        :param bidirectional:
        :param pretrained: 'flag to use or not pretrained embeddings'
        :param embedding_layer: 'pretrained embedding layer with glove embeddings'
        '''

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.cuda = cuda
        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.word_dropout_type=word_dropout_type

        if pretrained == False:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        else:
            self.embedding = embedding_layer
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.attribute_size = attribute_size
        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()
        self.enc_bidirectional = True
        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        print("latent_size: ", latent_size)
        print("attribute_size: ", attribute_size)
        print("hidden_size: ", hidden_size)
        print("hidden_factor : ", self.hidden_factor)
        self.latent2hidden = nn.Linear(latent_size + attribute_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)
        self.back = back


    def encoder(self, input_embedding, sorted_lengths, batch_size):
        '''

        The probabilistic encoder for the SVAE

        :param input_embedding:
        :param sorted_lengths:
        :param batch_size:
        :return: latent space, the mean and the log variance
        '''
        # ENCODER
        padded = rnn_utils.pad_sequence(input_embedding, batch_first=True)
        sorted_batch_lengths = [len(x) for x in padded]
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_batch_lengths, batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)



        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()



        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]), self.cuda)
        z = z * std + mean

        return z, mean, logv

    def word_dropout(self, step, maximum=0.7, minimum=0.3, warmup=2000, period=500):

        '''

        This is the Cyclical Word-Dropout

        :param step: current training step
        :param maximum: maximum word_dropout values
        :param minimum: minimum word_Dropout values
        :param warmup: number of training step before starting the cyclic
        :param period: periodo of the word_Dropout
        :return: word_dropout_rate
        '''

        if step < warmup:
            return maximum
        y = np.abs(np.cos(((2 * np.pi) / period) * step))
        if y > maximum:
            y = maximum
        if y < minimum:
            y = minimum
        return y

    def decoder(self, input_sequence, z , input_embedding, sorted_lengths, batch_size, word_dropout_rate):

        '''

        :param input_sequence: input sentence used for teacher forcing
        :param z: latent space
        :param input_embedding:
        :param sorted_lengths:
        :param batch_size:
        :param word_dropout_rate:
        :return: probability distribution over the vocabulary
        '''

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob = prob.to(self.cuda)
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        return outputs


    def forward(self, input_sequence, length, label, step,encoder=False):
        # PARAMETERS
        batch_size = input_sequence.size(0)

        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]
        label = label[sorted_idx]

        #EMBEDDING LAYER
        input_embedding = self.embedding(input_sequence)
        # ENCODER

        z, mean, logv = self.encoder(input_embedding, sorted_lengths, batch_size)

        # TODO : ReCon loss troubleshooting - remove after resolution
        # what are input id's from forward - Bert2LatentConnector
        # are they same as input_sequence ?
        #print("input_sequence: ", input_sequence)
        #print(" z from CGA : comparing with latent_z from Causal Lens: ", z)

        # TODO : SP : Re-Construction Loss
        """
        attention_mask = (inputs > 0).float()
        # logger.info(inputs)
        # logger.info(attention_mask)
        # logger.info(labels)
        reconstrution_mask = (labels != 50257).float()  # 50257 is the padding token for GPT2
        sent_length = torch.sum(reconstrution_mask, dim=1)

        # BERTForLatentConnector Forward function which is the Encoder 
        def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)

            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape bsz x n_heads x N x N
            # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
            # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
                head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
            else:
                head_mask = [None] * self.config.num_hidden_layers

            embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
            encoder_outputs = self.encoder(embedding_output,
                                           extended_attention_mask,
                                           head_mask=head_mask)
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output)

            outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here

            return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

        # Connect hidden feature to the latent space
        latent_z, loss_kl = self.connect(pooled_hidden_fea)
        latent_z = latent_z.squeeze(1)


        # Decoding
        outputs = self.decoder(input_ids=labels, past=latent_z, labels=labels, label_ignore=self.pad_token_id)
        loss_rec = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

        # GPT2ForLatentConnector Forward function which is the decoder 
        def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, label_ignore=None):


            transformer_outputs = self.transformer(input_ids,
                                                   past=past,
                                                   attention_mask=attention_mask,
                                                   token_type_ids=token_type_ids,
                                                   position_ids=position_ids,
                                                   head_mask=head_mask, 
                                                   latent_as_gpt_emb=self.latent_as_gpt_emb,
                                                   latent_as_gpt_memory=self.latent_as_gpt_memory)
            hidden_states = transformer_outputs[0]

            lm_logits = self.lm_head(hidden_states)

            outputs = (lm_logits,) + transformer_outputs[1:]
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(ignore_index=label_ignore, reduce=False) # 50258 is the padding id, otherwise -1 is used for masked LM.
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1))
                loss = torch.sum(loss.view(-1, shift_labels.shape[-1]), -1)
                outputs = (loss,) + outputs
            return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

        """

        if encoder:
            return z, label, input_sequence
        # DECODER
        z_a = torch.cat((z, label), dim=1)


        if self.word_dropout_type == 'static':
            word_dropout_rate = self.word_dropout_rate
        if self.word_dropout_type == 'cyclical':
            word_dropout_rate = self.word_dropout(step)


        outputs = self.decoder(input_sequence, z_a, input_embedding, sorted_lengths, batch_size, word_dropout_rate)
        # OUTPUT FORMAT
        padded_outputs = self.formatting_output(outputs, sorted_idx)
        b,s,_ = padded_outputs.size()
        # PROJECTION TO VOCAB
        out = self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2)))
        back_prob = nn.functional.softmax(out, dim=1)
        logp = nn.functional.log_softmax(out, dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        """
        if self.back:
            back_prob = back_prob.view(b, s, self.embedding.num_embeddings)
            back_input = torch.argmax(back_prob, dim=2)
            back_input = back_input[sorted_idx]
            back_input_embedding = self.embedding(back_input)
            z_back, mean_back, logv_back = self.encoder(back_input_embedding, sorted_lengths, batch_size)
            loss = torch.abs(z - z_back)
            print("type of loss: ", type(loss))
            print("shape of loss: ", loss.size())
            l1_loss = loss.sum()/batch_size
            return logp, mean, logv, z, z_a,l1_loss
    
        if self.back: # TODO : loss_rec
            # loss_rec=0
            back_prob = back_prob.view(b, s, self.embedding.num_embeddings)
            back_input = torch.argmax(back_prob, dim=2)
            back_input = back_input[sorted_idx]
            back_input_embedding = self.embedding(back_input)
            z_back, mean_back, logv_back = self.encoder(back_input_embedding, sorted_lengths, batch_size)
            # print("z_back size: ", z_back.size())
            # print("z size: ", z.size())
            # print("z_back : ", z_back)
            #z_back_normal = normalize(z_back, p=2.0, dim=0)
            # mins = z_back.min(dim=1, keepdim=True)
            # maxs = z_back.max(dim=1, keepdim=True)
            # print("mins: ", mins)
            # print("maxs: ", maxs)
            # z_back_normal = (z_back - mins) / (maxs - mins)
            z_back -= z_back.min(1, keepdim=True)[0]
            z_back /= z_back.max(1, keepdim=True)[0]
            # print("z_back_normal : ", z_back)
            loss_rec = F.binary_cross_entropy(z_back, z.detach())
            loss_rec_mean = loss_rec.sum()/batch_size
            print("loss_rec_mean: ", loss_rec_mean.size())
            print("loss_rec_mean: ", loss_rec_mean)
            return logp, mean, logv, z, z_a,loss_rec_mean
        """
        if self.back: # TODO : Context aware loss + loss_rec
            # loss_rec=0
            back_prob = back_prob.view(b, s, self.embedding.num_embeddings)
            back_input = torch.argmax(back_prob, dim=2)
            back_input = back_input[sorted_idx]
            back_input_embedding = self.embedding(back_input)
            z_back, mean_back, logv_back = self.encoder(back_input_embedding, sorted_lengths, batch_size)
            loss = torch.abs(z - z_back)
            l1_loss = loss.sum() / batch_size
            z_back -= z_back.min(1, keepdim=True)[0]
            z_back /= z_back.max(1, keepdim=True)[0]
            loss_rec = F.binary_cross_entropy(z_back, z.detach())
            loss_rec_mean = loss_rec.sum()/batch_size
            return logp, mean, logv, z, z_a,l1_loss, loss_rec_mean

        return logp, mean, logv, z, z_a, None

    def formatting_output(self, outputs, sorted_idx):
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)

        padded_outputs = padded_outputs[reversed_idx]
        return padded_outputs


    def inference(self, attribute_list, n = 2, z=None):

        '''

        :param attribute_list: concatenated hot-one representation of the attributes we want to control
        :param n: number of sentences we want to generate
        :param z: samples latent space
        :return: generated tokens for each sentence
        '''

        if z is None:
            batch_size = n
            z = to_var(torch.randn([1, self.latent_size]), "cpu")
            z_f = z
            for i in range(1):
                z_f = torch.cat((z_f, z), dim=0)
            '''
            if sentiment == "pres_pos":
                b = np.array([[0,1,1,0],[0,1,1,0]]).astype(np.float32)
            if sentiment == "past_pos":
                b = np.array([[0,1,0,1],[0,1,0,1]]).astype(np.float32)
            if sentiment == "pres_neg":
                b = np.array([[1, 0, 1, 0], [1, 0, 1, 0]]).astype(np.float32)
            if sentiment == "past_neg":
                b = np.array([[1, 0, 0, 1], [1, 0, 0, 1]]).astype(np.float32)

            '''
            #all positives


            if attribute_list == "pastpossingular":
                b = np.array([[1, 0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0,1]]).astype(np.float32)
            if attribute_list == "pastposplural":
                b = np.array([[0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0,1]]).astype(np.float32)
            if attribute_list == "prespossingular":
                b = np.array([[1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0,1]]).astype(np.float32)
            if attribute_list == "presposplural":
                b = np.array([[0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0,1]]).astype(np.float32)

            if attribute_list == "pastnegsingular":
                b = np.array([[1, 0, 0, 0, 1, 1, 0], [1, 0, 0, 0, 1, 1, 0]]).astype(np.float32)
            if attribute_list == "pastnegplural":
                b = np.array([[0, 0, 1, 0, 1, 1, 0], [0, 0, 1, 0, 1, 1, 0]]).astype(np.float32)
            if attribute_list == "presnegsingular":
                b = np.array([[1, 0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 0]]).astype(np.float32)
            if attribute_list == "presnegplural":
                b = np.array([[0, 0, 1, 1, 0, 1, 0], [0, 0, 1, 1, 0, 1, 0]]).astype(np.float32)

            # Attribute list for [ Positive, Negative, Singular, Neutral, Plural]
            """
                possingular -> 1 0 1 0 0 

                negsingular -> 0 1 1 0 0
                
                posneutral -> 1 0 0 1 0
                
                negneutral -> 0 1 0 1 0
                
                posplural -> 1 0 0 0 1
                
                negplural -> 0 1 0 0 1
            """
            if attribute_list == "possingular":
                b = np.array([[1, 0, 1, 0, 0], [1, 0, 1, 0, 0]]).astype(np.float32)

            if attribute_list == "negsingular":
                b = np.array([[0, 1, 1, 0, 0], [0, 1, 1, 0, 0]]).astype(np.float32)

            if attribute_list == "posneutral":
                b = np.array([[1, 0, 0, 1, 0], [1, 0, 0, 1, 0]]).astype(np.float32)

            if attribute_list == "negneutral":
                b = np.array([[0, 1, 0, 1, 0], [0, 1, 0, 1, 0]]).astype(np.float32)

            if attribute_list == "posplural":
                b = np.array([[1, 0, 0, 0, 1], [1, 0, 0, 0, 1]]).astype(np.float32)

            if attribute_list == "negplural":
                b = np.array([[0, 1, 0, 0, 1], [0, 1, 0, 0, 1]]).astype(np.float32)

            # For Tense , Sentiment and Pronoun

            if attribute_list == "PresPosPlural":
                b = np.array([[0, 0, 1, 0, 1, 1, 0], [0, 0, 1, 0, 1, 1, 0]]).astype(np.float32)
                #b = np.array([[0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 0, 1, 1, 0]]).astype(np.float32)
            if attribute_list == "PresPosSingular":
                b = np.array([[1, 0, 0, 0, 1, 1, 0], [1, 0, 0, 0, 1, 1, 0]]).astype(np.float32)
            if attribute_list == "PresPosNeutral":
                b = np.array([[0, 1, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 1, 0]]).astype(np.float32)

            if attribute_list == "PastPosPlural":
                b = np.array([[0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0, 1]]).astype(np.float32)
            if attribute_list == "PastPosSingular":
                b = np.array([[1, 0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0, 1]]).astype(np.float32)
            if attribute_list == "PastPosNeutral":
                b = np.array([[0, 1, 0, 0, 1, 0, 1], [0, 1, 0, 0, 1, 0, 1]]).astype(np.float32)

            if attribute_list == "PresNegPlural":
                b = np.array([[0, 0, 1, 1, 0, 1, 0], [0, 0, 1, 1, 0, 1, 0]]).astype(np.float32)
            if attribute_list == "PresNegSingular":
                b = np.array([[1, 0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 0]]).astype(np.float32)
            if attribute_list == "PresNegNeutral":
                b = np.array([[0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0]]).astype(np.float32)

            if attribute_list == "PastNegPlural":
                b = np.array([[0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 1]]).astype(np.float32)
            if attribute_list == "PastNegSingular":
                b = np.array([[1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 1]]).astype(np.float32)
            if attribute_list == "PastNegNeutral":
                b = np.array([[0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0, 1]]).astype(np.float32)


            c = torch.from_numpy(b)
            z = torch.cat((z_f, c), dim=1)

        else:
            batch_size = z.size(0)


        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        if self.bidirectional:
            hidden = hidden.squeeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t=0
        while(t<self.max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long(), "cpu")
            input_sequence = input_sequence.unsqueeze(1)
            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)
            if t == 0:
                l = self.outputs2vocab(output)

            logits = self.outputs2vocab(output)
            l = torch.cat((l,logits), dim=1)
            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z, l

    def inference_sent(self,sentiment,n = 2, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([1, self.latent_size]), "cpu")
            z_f = z
            for i in range(1):
                z_f = torch.cat((z_f, z), dim=0)

            if sentiment == "Negative":

                b = np.array([[0, 1], [0, 1]]).astype(np.float32)
            if sentiment == "Positive":
                b = np.array([[1, 0], [1, 0]]).astype(np.float32)
            c = torch.from_numpy(b)
            z = torch.cat((z_f, c), dim=1)

        else:
            batch_size = z.size(0)


        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        if self.bidirectional:
            hidden = hidden.squeeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t=0
        while(t<self.max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long(), "cpu")
            input_sequence = input_sequence.unsqueeze(1)
            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)
            if t == 0:
                l = self.outputs2vocab(output)

            logits = self.outputs2vocab(output)
            l = torch.cat((l,logits), dim=1)
            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z, l

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':

            #dist = torch.nn.functional.softmax(dist,1)
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
