from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput
import numpy as np
import torch
import torch.nn as nn

Qmc_num_labels=2
Tnews_num_labels=15

class BertForMultiTask(BertPreTrainedModel):

    def __init__(self, config, tnews_num_labels=Tnews_num_labels, qmc_num_labels=Qmc_num_labels):
        super().__init__(config)
        self.csc_num_labels = config.vocab_size
        self.tnews_num_labels = tnews_num_labels
        self.qmc_num_labels = qmc_num_labels
        ## hard-parameter: the shared layer
        self.bert = BertModel(config, add_pooling_layer=True)## add pooling layer (dense+tanh)
        ### classifier_dropout: The dropout ratio for the classification head.
        ### hidden_dropout_prob (float, optional, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        ### output layer for csc
        self.token_dropout = nn.Dropout(classifier_dropout)
        self.token_classifier = nn.Linear(config.hidden_size, self.csc_num_labels)
        ### outputlayer for tnews
        self.tnews_dropout = nn.Dropout(classifier_dropout)
        self.tnews_classifier = nn.Linear(config.hidden_size, self.tnews_num_labels)
        ## output_layer for qmc
        self.qmc_dropout = nn.Dropout(classifier_dropout)
        self.qmc_classifier = nn.Linear(config.hidden_size, self.qmc_num_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None, ##batch,seq
        attention_mask=None, ##batch,seq
        token_type_ids=None, ##batch,seq
        task_id=None, ##batch
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        loss_all=[]
        #print(labels.shape)
        ### csc task
        task_id_filter = task_id==1
        if task_id_filter.any():
            ####  (batch_size of task1, sequence_length, hidden_size)
            sequence_output = outputs[0][task_id_filter]
            sequence_output = self.token_dropout(sequence_output)
            logits = self.token_classifier(sequence_output) ####  (batch_size of task1, sequence_length, vocab_size)
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=0) # ignore padding
                loss = loss_fct(logits.view(-1, self.csc_num_labels), labels[task_id_filter].view(-1))
            loss_all.append(loss)

        ## qustion matching task
        task_id_filter = task_id==3
        if task_id_filter.any():
            #print(outputs[1].shape)
            pooled_output = outputs[1][task_id_filter] ##(batch_size of task3,hidden_size)
            pooled_output = self.qmc_dropout(pooled_output)
            logits = self.qmc_classifier(pooled_output) ##batch_size of task3,num_label
            loss = None
            if labels is not None:
                labels_seq=labels[:,0][task_id_filter] ##from (batch,seq) to (batch)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.qmc_num_labels), labels_seq.view(-1))
            loss_all.append(loss)

        ### tnews: text classification
        task_id_filter = task_id==2
        if task_id_filter.any():
            pooled_output = outputs[1][task_id_filter]  ##(batch_size,hidden_size)
            pooled_output = self.tnews_dropout(pooled_output)
            logits = self.tnews_classifier(pooled_output) ##batch_size,num_labels
            loss = None
            if labels is not None:
                labels_seq=labels[:,0][task_id_filter] ##from (batch,seq) to (batch)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.tnews_num_labels), labels_seq.view(-1))
            loss_all.append(loss)

        if loss_all:
            loss = torch.stack(loss_all) ## cancatenate all the loss
            #print(loss.get_device())
            # logits are only useful during eval when there is only one task, thus only one logits
            outputs = (loss.mean(),) + (logits,) + outputs
        return outputs

'''
task_csc = Task(1,'csc','task_classification')
task_tnews = Task(2, 'tnews', 'seq_classification')
task_qmc = Task(3,'afqmc','question-similarity')
'''

class MultiTaskReLM(nn.Module):

    def __init__(self, model, tokenizer, verbalizer_tnews, verbalizer_afqmc, prompt_length_sent, prompt_length_csc, linear_prob):
        super().__init__()
        self.config = model.config
        self.tokenizer = tokenizer
        self.prompt_length_csc = prompt_length_csc
        self.prompt_length_sent = prompt_length_sent
        self.verbalizer_tnews = verbalizer_tnews
        self.verbalizer_afqmc = verbalizer_afqmc
        self.tnews_label_words_ids = None if verbalizer_tnews == None else verbalizer_tnews.label_words_ids
        self.afqmc_label_words_ids = None if verbalizer_afqmc == None else verbalizer_afqmc.label_words_ids

        self.csc_num_labels = self.config.vocab_size
        self.tnews_num_labels = None if verbalizer_tnews == None else verbalizer_tnews.num_labels
        self.afqmc_num_labels = None if verbalizer_afqmc == None else verbalizer_afqmc.num_labels

        self.model = model  # mlm
        # the embdedding layer of BERT
        self.model_type = self.config.model_type.split("-")[0]
        self.word_embeddings = getattr(self.model, self.model_type).embeddings.word_embeddings
        
        # prompt embedding for afqmc
        self.prompt_embeddings_afqmc = nn.Embedding(self.prompt_length_sent, self.config.hidden_size)
        self.prompt_lstm_afqmc = nn.LSTM(input_size=self.config.hidden_size,
                                         hidden_size=self.config.hidden_size,
                                         num_layers=2,
                                         bidirectional=True,
                                         batch_first=True)
        self.prompt_linear_afqmc = nn.Sequential(nn.Linear(2 * self.config.hidden_size, self.config.hidden_size),
                                                 nn.ReLU(),
                                                 nn.Linear(self.config.hidden_size, self.config.hidden_size))

        # prompt embedding for tnews
        self.prompt_embeddings_tnews = nn.Embedding(self.prompt_length_sent, self.config.hidden_size)
        self.prompt_lstm_tnews = nn.LSTM(input_size=self.config.hidden_size,
                                         hidden_size=self.config.hidden_size,
                                         num_layers=2,
                                         bidirectional=True,
                                         batch_first=True)
        self.prompt_linear_tnews = nn.Sequential(nn.Linear(2 * self.config.hidden_size, self.config.hidden_size),
                                                 nn.ReLU(),
                                                 nn.Linear(self.config.hidden_size, self.config.hidden_size))

        # prompt embedding for csc
        self.prompt_embeddings_csc = nn.Embedding(2*self.prompt_length_csc, self.config.hidden_size)
        # LSTM: input:(batch,seq,input_size)-->output[0]:(batch,seq,2*hidden)
        self.prompt_lstm_csc = nn.LSTM(input_size=self.config.hidden_size,
                                       hidden_size=self.config.hidden_size,
                                       num_layers=2,
                                       bidirectional=True,
                                       batch_first=True)
        self.prompt_linear_csc = nn.Sequential(nn.Linear(2 * self.config.hidden_size, self.config.hidden_size),
                                               nn.ReLU(),
                                               nn.Linear(self.config.hidden_size, self.config.hidden_size))
        if linear_prob:
            self.classifier = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        prompt_mask=None,
        active_bits=None,
        task_id=None, ## batch
        labels=None,
        inputs_embeds=None,
        output_hidden_states=True,
        return_dict=True,
        apply_prompt=True,
        linear_prob=False
    ):
        csc_task_filter = (task_id == 1)
        tnews_task_filter = (task_id == 2)
        afqmc_task_filter = (task_id == 3)
        # get embdding of all the tasks
        inputs_embeds = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        if apply_prompt:

            # afqmc
            replace_embeds_afqmc = self.prompt_embeddings_afqmc(torch.LongTensor(list(range(self.prompt_length_sent))).to(inputs_embeds.device))
            replace_embeds_afqmc = replace_embeds_afqmc.unsqueeze(0)
            replace_embeds_afqmc = self.prompt_lstm_afqmc(replace_embeds_afqmc)[0]  # (prompt_length,2*hidden_size)
            replace_embeds_afqmc = self.prompt_linear_afqmc(replace_embeds_afqmc).squeeze()  # (prompt_length,hidden)
            # tnews
            replace_embeds_tnews = self.prompt_embeddings_tnews(torch.LongTensor(list(range(self.prompt_length_sent))).to(inputs_embeds.device))
            replace_embeds_tnews = replace_embeds_tnews.unsqueeze(0)
            replace_embeds_tnews = self.prompt_lstm_tnews(replace_embeds_tnews)[0]  # (prompt_length,2*hidden_size)
            replace_embeds_tnews = self.prompt_linear_tnews(replace_embeds_tnews).squeeze()  # (prompt_length,hidden)
            # csc
            replace_embeds_csc = self.prompt_embeddings_csc(torch.LongTensor(list(range(2*self.prompt_length_csc))).to(input_ids.device))
            replace_embeds_csc = replace_embeds_csc.unsqueeze(0)  # (1,2*prompt_length,hidden_size)
            replace_embeds_csc = self.prompt_lstm_csc(replace_embeds_csc)[0]  # (2*prompt_length,2*hidden_size)
            replace_embeds_csc = self.prompt_linear_csc(replace_embeds_csc).squeeze()  # (2*prompt_length,hidden_size)

            # prompt_mask (batch,seq)
            # (batch size for csc,seq)
            prompt_mask_csc = prompt_mask[csc_task_filter]
            blocked_indices_csc = (prompt_mask_csc == 1).nonzero().reshape((prompt_mask_csc.shape[0], 2*self.prompt_length_csc, 2))[:, :, 1]  # (batch size for csc,2*prompt_length_csc)
            # (batch size for tnews,seq)
            prompt_mask_tnews = prompt_mask[tnews_task_filter]
            blocked_indices_tnews = (prompt_mask_tnews == 1).nonzero().reshape((prompt_mask_tnews.shape[0], self.prompt_length_sent, 2))[:, :, 1]  # (batch size for tnews,prompt_length_sent)
            # (batch size for afqmc,seq)
            prompt_mask_afqmc = prompt_mask[afqmc_task_filter]
            blocked_indices_afqmc = (prompt_mask_afqmc == 1).nonzero().reshape((prompt_mask_afqmc.shape[0], self.prompt_length_sent, 2))[:, :, 1]  # (batch size for afqmc,prompt_length_sent)

            # replace the prompt positions in input_embeds with prompt embeddings correspondingly
            csc_i, tnews_i, afqmc_i = 0, 0, 0
            for i in range(inputs_embeds.shape[0]):
                if task_id[i] == 1:
                    for j in range(blocked_indices_csc.shape[1]):
                        inputs_embeds[i, blocked_indices_csc[csc_i, j],:] = replace_embeds_csc[j, :]
                    csc_i += 1
                elif task_id[i] == 2:
                    for j in range(blocked_indices_tnews.shape[1]):
                        inputs_embeds[i, blocked_indices_tnews[tnews_i,j], :] = replace_embeds_tnews[j, :]
                    tnews_i += 1
                else:
                    assert task_id[i] == 3
                    for j in range(blocked_indices_afqmc.shape[1]):
                        inputs_embeds[i, blocked_indices_afqmc[afqmc_i,j], :] = replace_embeds_afqmc[j, :]
                    afqmc_i += 1

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        logits = outputs.logits  # batch,seq,vocab_size
        loss_all = []
        # csc
        if csc_task_filter.any():
            csc_logits = logits[csc_task_filter]
            csc_loss = None
            if labels is not None:
                labels_csc = labels[csc_task_filter]
                input_csc = input_ids[csc_task_filter]
                labels_csc[input_csc == labels_csc] = -100
                loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
                csc_loss = loss_fct(csc_logits.view(-1, self.csc_num_labels), labels_csc.view(-1))
            logits_output = csc_logits # batch,seq,vocab_size
            loss_all.append(csc_loss)
        # tnews
        if tnews_task_filter.any():
            mask_length = 2
            if linear_prob:
                hidden_states = outputs.hidden_states
                logits = self.classifier(hidden_states[-1])
            tnews_logits = logits[tnews_task_filter]  # tnews_batch,seq,vocab
            # tnews_batch,seq
            tnews_active_bits = active_bits[tnews_task_filter]
            tnews_logits = tnews_logits[torch.where(tnews_active_bits != -100)]\
                .view(-1, mask_length, self.tokenizer.vocab_size)  # tnews_batch,mask_length=2,vocab

            # tnews_batch,num_label
            label_words_logits_1 = tnews_logits[:,0, self.tnews_label_words_ids[:, 0]]
            # tnews_batch,num_label
            label_words_logits_2 = tnews_logits[:,1, self.tnews_label_words_ids[:, 1]]
            label_words_logits = label_words_logits_1 * label_words_logits_2
            assert label_words_logits.shape[-1] == self.tnews_num_labels
            tnews_loss = None
            if labels is not None:
                ## batch,seq-->batch
                labels_tnews = labels[tnews_task_filter][:,0]
                loss_fct = nn.CrossEntropyLoss()
                tnews_loss = loss_fct(label_words_logits.view(-1, self.tnews_num_labels), labels_tnews.view(-1))
            logits_output = label_words_logits # afqmc_batch,num_label
            loss_all.append(tnews_loss)
        # afqmc
        if afqmc_task_filter.any():
            afqmc_logits = logits[afqmc_task_filter]  # afqmc_batch,seq,vocab
            # afqmc_batch,seq
            afqmc_active_bits = active_bits[afqmc_task_filter]
            afqmc_logits = afqmc_logits[torch.where(afqmc_active_bits != -100)]  # afqmc_batch,vocab

            label_words_logits = afqmc_logits[:, self.afqmc_label_words_ids] # afqmc_batch,num_label,num_label_mapping
            label_words_logits = torch.sum(label_words_logits, dim=-1)  # afqmc_batch,num_label
            afqmc_loss = None
            if labels is not None:
                labels_afqmc = labels[afqmc_task_filter][:,0]
                loss_fct = nn.CrossEntropyLoss()
                afqmc_loss = loss_fct(label_words_logits.view(-1, self.afqmc_num_labels), labels_afqmc.view(-1))
            logits_output = label_words_logits # afqmc_batch,num_label
            loss_all.append(afqmc_loss)
        loss = torch.stack(loss_all).mean() ## cancatenate all the loss
        if output_hidden_states:
            return loss, logits_output, outputs.hidden_states
        return loss, logits_output
        