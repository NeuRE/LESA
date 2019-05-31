import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.layers as layers
from tqdm import tqdm
import time
from modules import *
import df_test2 as modeltest
import os
import sys
import horovod.tensorflow as hvd
import re_output_fns as test

# os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1) 
random.seed(1)
tf.logging.set_verbosity(tf.logging.INFO)
#random.seed(1)

class Settings(object):
    def __init__(self):
        self.vocab_size = 154995
        self.len_sentence = 40
        self.num_epochs = 100
        self.num_classes = 19
        self.cnn_size = 230
        self.num_layers = 5
        self.word_embedding = 200
        self.fea_dim = 200
        self.pos_size = 5
        self.pos_num = 123
        self.keep_prob = 0.5
        self.batch_size = 20
        self.num_steps = 10000
        self.lr = 0.0005
        self.num_blocks = 2
        self.num_heads = 4
        self.sinusoid = False
        self.is_training = True
        self.dropout_rate = 0.5
        self.num_entity_classes = 4
        self.father_node_layer = 0
        self.relation_to_father_species = 38
        self.pos_species = 34
        self.pos_layer = 1


class CNN():
    def __init__(self, word_embeddings, setting,lr,num_blocks,num_heads, gamma1, gamma2,gamma3,gamma4,gamma5,gamma6,len_sentence):

        self.vocab_size = setting.vocab_size
        self.len_sentence = len_sentence
        self.num_epochs = setting.num_epochs
        self.num_classes = num_classes = setting.num_classes
        self.cnn_size = setting.cnn_size
        self.num_layers = setting.num_layers
        self.pos_size = setting.pos_size
        self.pos_num = setting.pos_num
        self.word_embedding = setting.word_embedding
        #self.lr = lr
        self.fea_dim = setting.fea_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.sinusoid = setting.sinusoid
        self.is_training = setting.is_training
        self.num_entity_classes = setting.num_entity_classes
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gamma4 = gamma4
        self.gamma5 = gamma5
        self.gamma6 = gamma6
        self.father_node_layer = self.num_blocks - 1
        self.relation_to_father_species = setting.relation_to_father_species
        self.pos_species = setting.pos_species
        self.pos_layer = self.num_blocks - 2
        self.entity_layer = self.num_blocks - 1


      #  word_embedding = tf.get_variable(initializer=word_embeddings, name='word_embedding')
        word_embedding = word_embeddings
        pos1_embedding = tf.get_variable('pos1_embedding', [self.pos_num, self.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding', [self.pos_num, self.pos_size])
        # relation_embedding = tf.get_variable('relation_embedding', [self.num_classes, self.cnn_size])

        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_word')

        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_pos2')

        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32)
        self.dropout_rate = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)
        self.entity_info = tf.placeholder(tf.int32, [None, len_sentence], name='entity_info')
        # combine
        # self.input_dependency = tf.placeholder(dtype=tf.float32, shape=[None,len_sentence], name='input_dependency')
        # with tf.name_scope('input'):
        # image = tf.placeholder(tf.float32, [None, len_sentence, word_embedding], name='feature')
        self.father_node = tf.placeholder(tf.int32, [None, len_sentence], name='father_node')  # 父节点
        self.relation_to_father = tf.placeholder(tf.int32, [None, len_sentence], name='relation_to_father')  # 与父节点关系
        # self.tokens_to_keep = tf.placeholder(tf.float32, [None, len_sentence], name='tokens_to_keep') #mask
        self.pos = tf.placeholder(tf.int32, [None, len_sentence], name='pos')
        self.input_word_ebd = tf.nn.embedding_lookup(word_embedding, self.input_word)  # image

        self.input_pos1_ebd = tf.nn.embedding_lookup(pos1_embedding, self.input_pos1)
        self.input_pos2_ebd = tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)
        # with tf.device('/gpu:1'):
        with tf.variable_scope("encoder"):
            ## Embedding
            self.enc = tf.layers.dense(self.input_word_ebd, self.fea_dim)
            print(self.enc.get_shape)
            ## Positional Encoding
            if self.sinusoid:
                self.enc += positional_encoding(self.input_word,
                                                num_units=self.fea_dim,
                                                zero_pad=False,
                                                scale=False,
                                                scope="enc_pe")
            else:
                self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_word)[1]), 0), [tf.shape(self.input_word)[0], 1]),
                                      vocab_size=self.len_sentence,
                                      num_units=self.fea_dim,
                                      zero_pad=False,
                                      scale=False,
                                      scope="enc_pe")

            ## Dropout
            self.enc = tf.layers.dropout(self.enc,
                                         rate=self.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))

            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(queries=self.enc,
                                                   keys=self.enc,
                                                   num_units=None,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=False)

                    # if i == self.num_blocks - 1:
                    # self.special_enc = tf.split(self.enc,self.num_heads,axis = 2)
                    # self.special_head = self.special_enc[0]

                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4 * self.fea_dim, self.fea_dim])
                    if i == 0:
                        self.encoder_output = self.enc
                    else:
                        self.encoder_output += self.enc
                    if i == self.father_node_layer:
                        feature = self.enc
                    if i == self.pos_layer:
                        pos_feature = self.enc
                    if i == self.entity_layer:
                        entity_feature = self.enc
        # father_relation and father node predict
        batch_shape = tf.shape(self.father_node)
        batch_size = batch_shape[0]
        seq_len = batch_shape[1]

        keep_parent = tf.where(tf.equal(self.father_node, -2), tf.zeros([batch_size, seq_len], dtype=tf.int32), tf.ones([batch_size, seq_len], dtype=tf.int32))
        mask_parent = tf.multiply(keep_parent, self.father_node)
        output0 = test.parse_bilinear(feature, mask_parent, tf.cast(keep_parent, tf.float32))
        # output0 = test.parse_bilinear(image, self.father_node, self.tokens_to_keep)

        keep_relation = tf.where(tf.equal(self.relation_to_father, -2), tf.zeros([batch_size, seq_len], dtype=tf.int32), tf.ones([batch_size, seq_len], dtype=tf.int32))
        mask_relation = tf.multiply(keep_relation, self.relation_to_father)
        output1 = test.conditional_bilinear(mask_relation, self.relation_to_father_species, tf.cast(keep_relation, tf.float32), output0['dep_rel_mlp'], output0['head_rel_mlp'], mask_parent)

        keep_pos = tf.where(tf.equal(self.pos, -2), tf.zeros([batch_size, seq_len], dtype=tf.int32), tf.ones([batch_size, seq_len], dtype=tf.int32))
        mask_pos = tf.multiply(keep_pos, self.pos)
        output2 = test.pos_softmax_classifier(pos_feature, mask_pos, self.pos_species, tf.cast(keep_pos, tf.float32))
        
        keep_entity = tf.where(tf.equal(self.entity_info, -2), tf.zeros([batch_size, seq_len], dtype=tf.int32), tf.ones([batch_size, seq_len], dtype=tf.int32))
        mask_entity = tf.multiply(keep_entity, self.entity_info)
        output3 = test.ner_softmax_classifier(entity_feature, mask_entity, 45, tf.cast(keep_entity, tf.float32))

        self.loss0 = output0['loss']
        self.loss1 = output1['loss']
        self.loss2 = output2['loss']
        self.loss3 = output3['loss']

        # self.encoder_output = tf.expand_dims(self.encoder_output,-1)

        self.inputs = tf.concat(axis=2, values=[self.enc, self.input_pos1_ebd, self.input_pos2_ebd])
        print('=---------------')
        print(self.inputs)
        self.inputs = tf.reshape(self.inputs, [-1, self.len_sentence, self.word_embedding + self.pos_size * 2, 1])

        conv = layers.conv2d(inputs=self.inputs, num_outputs=self.cnn_size, kernel_size=[3, 210], stride=[1, 210], padding='SAME')

        max_pool = layers.max_pool2d(conv, kernel_size=[self.len_sentence, 1], stride=[1, 1])
        self.sentence = tf.reshape(max_pool, [-1, self.cnn_size])

        tanh = tf.nn.tanh(self.sentence)
        drop = layers.dropout(tanh, keep_prob=self.keep_prob)

        self.outputs = layers.fully_connected(inputs=drop, num_outputs=self.num_classes, activation_fn=tf.nn.softmax)

        '''
        self.y_index =  tf.argmax(self.input_y,1,output_type=tf.int32)
        self.indexes = tf.range(0, tf.shape(self.outputs)[0]) * tf.shape(self.outputs)[1] + self.y_index
        self.responsible_outputs = - tf.reduce_mean(tf.log(tf.gather(tf.reshape(self.outputs, [-1]),self.indexes)))
        '''
        # loss 
        self.cross_loss = -tf.reduce_mean(tf.log(tf.reduce_sum(self.input_y * self.outputs, axis=1)))
        self.reward = tf.log(tf.reduce_sum(self.input_y * self.outputs, axis=1))

        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())
        

        self.l1_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l1_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())
        self.sum_loss = self.cross_loss + self.gamma2 * self.l2_loss + self.gamma1 * self.l1_loss
        self.final_loss = self.sum_loss + self.gamma3 * self.loss0 + self.gamma4 * self.loss1 + self.gamma5 * self.loss2 + self.gamma6 * self.loss3

        # accuracy
        self.pred = tf.argmax(self.outputs, axis=1)
        self.pred_prob = tf.reduce_max(self.outputs, axis=1)

        self.y_label = tf.argmax(self.input_y, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.y_label), 'float'))

        # Training Scheme
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        self.train_op = self.optimizer.minimize(self.final_loss, global_step=self.global_step)

        self.tvars = tf.trainable_variables()

        # manual update parameters
        self.tvars_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.tvars_holders.append(placeholder)

        self.update_tvar_holder = []
        for idx, var in enumerate(self.tvars):
            update_tvar = tf.assign(var, self.tvars_holders[idx])
            self.update_tvar_holder.append(update_tvar)


def train(path_train_word, path_train_pos1, path_train_pos2, path_train_y, path_father_node, path_relation_to_father, path_pos,path_entity_pair, save_path,lr, batch_size,num_blocks,num_heads,gamma1, gamma2,gamma3,gamma4,gamma5,gamma6,seed,trans_drop,cnn_drop,attenuation_rate,sen_len):
    print('reading wordembedding')
    wordembedding = np.load('../entity_type/df_data/def_vec.npy')
    dict_relation2id = {}
    with open('../entity_type/df_data/df_relation2id.txt','r',encoding='utf-8') as input:
        lines = input.readlines()

    for line in lines:
        line = line.strip().split('\t')
        relation = line[0]
        id = int (line[1])
        dict_relation2id[relation] = id

    print('reading training data')

    cnn_train_word = np.load(path_train_word)
    cnn_train_pos1 = np.load(path_train_pos1)
    cnn_train_pos2 = np.load(path_train_pos2)
    cnn_train_y = np.load(path_train_y)

    # cnn_dependency = np.load(path_dependency)[indices]
    cnn_father_node = np.load(path_father_node)
    cnn_relation_to_father = np.load(path_relation_to_father)
    cnn_pos = np.load(path_pos)
    cnn_entity_pair = np.load(path_entity_pair)

    settings = Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(cnn_train_y[0])
    settings.num_steps = len(cnn_train_word) // int(batch_size)

    config = tf.ConfigProto(allow_soft_placement=True)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    config.gpu_options.allow_growth = False

    #   config.gpu_options.visible_device_list = ['1']
    # with tf.device('/gpu:0'):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    with tf.Graph().as_default():
        sess = tf.Session(config=config)
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = CNN(word_embeddings=wordembedding, setting=settings, lr=lr,num_blocks =num_blocks ,num_heads=num_heads,gamma1=gamma1,gamma2=gamma2,gamma3=gamma3,gamma4 = gamma4,gamma5 = gamma5,gamma6=gamma6,len_sentence=sen_len)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            start_lr = lr
            attenuation_rate = attenuation_rate
            min_lr = 0.00005
            best_f1 = 0
            # saver.restore(sess,save_path=save_path)
            for epoch in range(1, settings.num_epochs + 1):
                current_lr = start_lr
                if current_lr >= min_lr:
                    start_lr = current_lr * attenuation_rate
                filestr = "../entity_type/df_data/df_train_output2.txt"
                fw = open(filestr,'w',encoding='utf-8')
                bar = tqdm(range(settings.num_steps), desc='epoch {}, loss=0.000000, accuracy=0.000000,sum_loss=0.000000,loss0=0.000000,loss1=0.000000,loss2=0.000000'.format(epoch))
                index_list = [i for i in range(len(cnn_train_y))]
                random.seed(seed)
                random.shuffle(index_list)
                #            print(index)
                batch_index = 0
                for _ in bar:
                    sample_list = []
                    # sample_list = index_list[0:10]
                    # print(index_list[0:10])
                    # sample_list = random.sample(range(len(cnn_train_y)),settings.batch_size)
                    # print(str(settings.batch_size * batch_index)+"\t"+str(settings.batch_size * (batch_index+1)))
                    sample_list = index_list[(batch_size * batch_index):(batch_size * (batch_index + 1))]
#                    batch_index = batch_index + 1
    #                print(batch_index)
                    batch_train_word = [cnn_train_word[x] for x in sample_list]

                    batch_train_y = [cnn_train_y[x] for x in sample_list]
                    batch_train_pos1 = [cnn_train_pos1[x] for x in sample_list]
                    batch_train_pos2 = [cnn_train_pos2[x] for x in sample_list]
                    # batch_dependency = [cnn_dependency[x] for x in sample_list] 
                    batch_father_node = [cnn_father_node[x] for x in sample_list]
                    batch_relation_to_father = [cnn_relation_to_father[x] for x in sample_list]
                    batch_pos = [cnn_pos[x] for x in sample_list]
                    batch_entity = [cnn_entity_pair[x] for x in sample_list]

                    entity_info = []
                    for index in range(batch_size):
                        pos1_list = batch_train_pos1[index].tolist()
                        pos2_list = batch_train_pos2[index].tolist()
                        pos_list = [-2 for i in range(sen_len)]
                        if 61 in pos1_list:
                            pos_list[pos1_list.index(61)] = batch_entity[index][0]
                        if 61 in pos2_list:
                            pos_list[pos2_list.index(61)] = batch_entity[index][1]
                        entity_info.append(pos_list)             
 
                    entity1_category = []
                    entity2_category = []

                    feed_dict = {}
                    feed_dict[model.input_word] = batch_train_word
                    feed_dict[model.input_pos1] = batch_train_pos1
                    feed_dict[model.input_pos2] = batch_train_pos2
                    feed_dict[model.input_y] = batch_train_y
                    feed_dict[model.keep_prob] = 1-cnn_drop
                    feed_dict[model.dropout_rate] = trans_drop
                    # feed_dict[model.input_dependency] = batch_dependency
                    feed_dict[model.father_node] = batch_father_node
                    feed_dict[model.relation_to_father] = batch_relation_to_father
                    feed_dict[model.pos] = batch_pos
                    feed_dict[model.lr] = current_lr
                    feed_dict[model.entity_info] = entity_info
                    # print(batch_dependency)
                    _, loss, step, sum_loss, loss0, loss1, loss2 ,relation = sess.run([model.train_op, model.final_loss, model.accuracy, model.sum_loss, model.loss0, model.loss1, model.loss2,model.pred], feed_dict=feed_dict)
                    #filestr = "../entity_type/df_data/df_test_output.txt"
                    for i in range(len(sample_list)):
                        for rel in dict_relation2id.keys():
                            if dict_relation2id[rel] == relation[i]:
                                fw.write(str(sample_list[i])+"\t"+str(rel)+"\n")
                    batch_index = batch_index + 1
                    bar.set_description('epoch {} loss={:.6f} step={:.6f} sum_loss={:.6f},loss0={:.6f},loss1={:.6f},loss2={:.6f}'.format(epoch, loss, step, sum_loss, loss0, loss1, loss2))
                save_path = '../entity_type/df_model2/model_' + str(gamma1) + '_' + str(gamma2) + 'epoch' + str(epoch) + '_new_.ckpt'
                saver.save(sess, save_path=save_path)
                fw.close()
                os.system('perl semeval2010_task8_scorer-v1.2_test.pl ../entity_type/df_data/df_train_input.txt ../entity_type/df_data/df_train_output2.txt')
                #f1 = os.system('perl semeval2010_task8_scorer-v1.2_test.pl ../entity_type/df_data/df_train_input.txt').split(' ')[4]
                result = 'result2.txt'
                file_result = open(result,'r')
                new_result = file_result.readlines()[-1]
                f1 = float(new_result.split(' ')[4])
                if f1 > best_f1:
                    best_f1 = f1        
                modeltest.produce_pred_data(save_path=save_path, output_path='../entity_type/result/df_origin_pred_entitypair.pkl',lr=lr,batch_size=batch_size,num_blocks=num_blocks,num_heads=num_heads,gamma1=gamma1,gamma2=gamma2,gamma3=gamma3,gamma4 = gamma4,sen_len =sen_len )
                os.system('rm ../entity_type/df_model2/*')
                if (best_f1 < 82 and epoch >= 40) or (best_f1 < 81.5 and epoch >= 30) or (best_f1 < 80 and epoch >= 20):
                    break
                if f1 >=84.8:
                    save_path = '../entity_type/bestmodel/model_' + str(f1) +"_"+ 'epoch' + str(epoch) + '_new_.ckpt'
                    saver.save(sess, save_path=save_path)
                # result = modeltest.P_N(label_path = '../entity_type/df_data/label_entitypair.pkl',pred_path ='../entity_type/result/df_origin_pred_entitypair.pkl')
                # print('origin_cnn_P@100,200,300:', result)


class interaction():
    def __init__(self, sess, save_path='../entity_type/df_model/model.ckpt3'):
        self.settings = Settings()
        wordembedding = np.load('../entity_type/df_data/def_vec.npy').astype('float32')
        # typeembedding = np.load('../entity_type/df_data/type_embedding.npy').astype('float32')
        self.settings.is_training = False
        self.sess = sess
        with tf.variable_scope("model"):
            self.model = CNN(word_embeddings=wordembedding, setting=self.settings, lr=0.0005,num_blocks = 2,num_heads=2,gamma1=0.2,gamma2=0.1,gamma3=0.3,gamma4 = 0.5,gamma5=0.1,gamma6=0.1,len_sentence = 70)

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, save_path)

        self.train_word = np.load('../entity_type/df_data_40/def_train_word.npy')
        self.train_pos1 = np.load('../entity_type/df_data_40/def_train_pos1.npy')
        self.train_pos2 = np.load('../entity_type/df_data_40/def_train_pos2.npy')
        self.y_train = np.load('../entity_type/df_data_40/def_train_label.npy')
        # entity_type/cnndata/cnn_train_entity1.npy

    def test(self, batch_test_word, batch_test_pos1, batch_test_pos2):
        feed_dict = {}
        feed_dict[self.model.input_word] = batch_test_word
        feed_dict[self.model.input_pos1] = batch_test_pos1
        feed_dict[self.model.input_pos2] = batch_test_pos2
        feed_dict[self.model.keep_prob] = 1
        feed_dict[self.model.dropout_rate] = 0

        relation, prob = self.sess.run([self.model.pred, self.model.pred_prob], feed_dict=feed_dict)

        return (relation, prob)


if __name__ == '__main__':
    lr = float(sys.argv[1])
    batch_size = int(sys.argv[2])
    num_blocks = int(sys.argv[3])
    num_heads = int(sys.argv[4])
    gamma1 = float(sys.argv[5])
    gamma2 = float(sys.argv[6])
    gamma3 = float(sys.argv[7])
    gamma4 = float(sys.argv[8])
    gamma5 = float(sys.argv[9])
    gamma6 = float(sys.argv[10])
    seed = float(sys.argv[11])
    trans_drop = float(sys.argv[12])
    cnn_drop = float(sys.argv[13])
    attenuation_rate = float(sys.argv[14])
    sen_len = int(sys.argv[15])

    model_name = '../entity_type/df_model/cnnmodel_union_transformer_with_cnn_with_entity_desc_' + str(gamma1) + '_' + str(gamma2) + 'new_.ckpt'
    print(model_name)

    # train model
    print('train model')
    train('../entity_type/df_data_'+str(sen_len)+'/def_train_word.npy', '../entity_type/df_data_'+str(sen_len)+'/def_train_pos1.npy', '../entity_type/df_data_'+str(sen_len)+'/def_train_pos2.npy', '../entity_type/df_data_'+str(sen_len)+'/def_train_label.npy', \
          '../entity_type/df_data_'+str(sen_len)+'/def_train_parent.npy', '../entity_type/df_data_'+str(sen_len)+'/def_train_relation_to_parent.npy', '../entity_type/df_data_'+str(sen_len)+'/def_train_pos.npy', '../entity_type/df_data_'+str(sen_len)+'/train_e_type.npy',\
          model_name,lr, batch_size,num_blocks,num_heads,gamma1, gamma2,gamma3,gamma4,gamma5,gamma6,seed,trans_drop,cnn_drop,attenuation_rate,sen_len)

    # #get embedding
    # print('get_embedding')
    # get_sentence_embedding('cnndata/cnn_train_word.npy','cnndata/cnn_train_y.npy', 'model/origin_cnn_model.ckpt')
