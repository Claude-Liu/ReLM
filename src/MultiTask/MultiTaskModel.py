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
        