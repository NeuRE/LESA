import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.layers as layers
from tqdm import tqdm
import time
from modules import *
import test_entitytype as modeltest
import os 
import sys
import horovod.tensorflow as hvd
import re_output_fns as test


# os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

tf.logging.set_verbosity(tf.logging.INFO)

class Settings(object):
    def __init__(self):
        self.vocab_size = 154995
        self.len_sentence = 70
        self.num_epochs = 50
        self.num_classes = 53
        self.cnn_size = 230
        self.num_layers = 10
        self.word_embedding = 50
        self.fea_dim = 50
        self.pos_size = 5
        self.pos_num = 123
        self.keep_prob = 0.5
        self.batch_size = 100
        self.num_steps = 10000
        self.lr= 0.0005
        self.num_blocks = 6
        self.num_heads = 10
        self.sinusoid = False
        self.is_training = True
        self.dropout_rate = 0.1
        self.num_entity_classes = 4
        self.father_node_layer = 3
        self.relation_to_father_species = 39
        self.pos_species = 38
        self.pos_layer = 2

class CNN():

    def __init__(self, word_embeddings,type_embeddings,setting,gamma1,gamma2):

        self.vocab_size = setting.vocab_size
        self.len_sentence= len_sentence = setting.len_sentence
        self.num_epochs = setting.num_epochs
        self.num_classes = num_classes =setting.num_classes
        self.cnn_size = setting.cnn_size
        self.num_layers = setting.num_layers
        self.pos_size = setting.pos_size
        self.pos_num = setting.pos_num
        self.word_embedding = setting.word_embedding
        self.lr = setting.lr
        self.fea_dim = setting.fea_dim
        self.num_blocks = setting.num_blocks
        self.num_heads = setting.num_heads
        self.sinusoid = setting.sinusoid
        self.is_training = setting.is_training
        self.dropout_rate = setting.dropout_rate
        self.num_entity_classes = setting.num_entity_classes
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.father_node_layer = setting.father_node_layer
        self.relation_to_father_species = setting.relation_to_father_species
        self.pos_species = setting.pos_species
        self.pos_layer = setting.pos_layer

        word_embedding = tf.get_variable(initializer=word_embeddings, name='word_embedding')
        type_embedding = tf.get_variable(initializer=type_embeddings, name='type_embedding')

        pos1_embedding = tf.get_variable('pos1_embedding', [self.pos_num, self.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding', [self.pos_num, self.pos_size])
        #relation_embedding = tf.get_variable('relation_embedding', [self.num_classes, self.cnn_size])

        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_word')

        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_pos2')
        self.entity1_position = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence, self.fea_dim], name='input_pos1')
        self.entity2_position = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence, self.fea_dim], name='input_pos2')
        self.entity1_category = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='input_entity1_category')
        self.entity2_category = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='input_entity2_category')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32)

        #combine
        #self.input_dependency = tf.placeholder(dtype=tf.float32, shape=[None,len_sentence], name='input_dependency')
        # with tf.name_scope('input'):
            # image = tf.placeholder(tf.float32, [None, len_sentence, word_embedding], name='feature')
        self.father_node = tf.placeholder(tf.int32, [None, len_sentence], name='father_node') #父节点
        self.relation_to_father = tf.placeholder(tf.int32, [None, len_sentence], name='relation_to_father') #与父节点关系
        self.tokens_to_keep = tf.placeholder(tf.float32, [None, len_sentence], name='tokens_to_keep') #mask
        self.pos = tf.placeholder(tf.int32, [None, len_sentence], name='pos')
        self.input_word_ebd = tf.nn.embedding_lookup(word_embedding, self.input_word)#image

        self.input_type1_ebd = tf.nn.embedding_lookup(type_embedding, self.entity1_category)
        self.input_type2_ebd = tf.nn.embedding_lookup(type_embedding, self.entity2_category)

        self.input_pos1_ebd = tf.nn.embedding_lookup(pos1_embedding, self.input_pos1)
        self.input_pos2_ebd = tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)
        #with tf.device('/gpu:1'):
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
                    
                    #if i == self.num_blocks - 1:
                        #self.special_enc = tf.split(self.enc,self.num_heads,axis = 2)
                        #self.special_head = self.special_enc[0]
                        
                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4*self.fea_dim, self.fea_dim])
                    if i == 0:
                        self.encoder_output = self.enc
                    else:
                        self.encoder_output += self.enc
                    if i == self.father_node_layer:
                        feature = self.encoder_output
                    if i == self.pos_layer:
                        pos_feature = self.encoder_output
                 


        #father_relation and father node predict
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

        self.loss0 = output0['loss'] 
        self.loss1 = output1['loss'] 
        self.loss2 = output2['loss']

        #self.encoder_output = tf.expand_dims(self.encoder_output,-1)

        self.inputs =  tf.concat(axis=2,values=[self.enc,self.input_pos1_ebd,self.input_pos2_ebd])
        self.inputs =  tf.concat(axis=1,values=[self.inputs,self.input_type1_ebd,self.input_type2_ebd])
        
        self.inputs = tf.reshape(self.inputs, [-1,self.len_sentence+2,self.word_embedding+self.pos_size*2,1] )

        
        conv = layers.conv2d(inputs =self.inputs, num_outputs = self.cnn_size, kernel_size = [3,60],stride=[1,60],padding='SAME')

     
        max_pool = layers.max_pool2d(conv, kernel_size = [72,1], stride=[1,1])
        self.sentence = tf.reshape(max_pool, [-1, self.cnn_size])

 
        tanh = tf.nn.tanh(self.sentence)
        drop = layers.dropout(tanh,keep_prob=self.keep_prob)

   
        self.outputs = layers.fully_connected(inputs=drop, num_outputs = self.num_classes,activation_fn = tf.nn.softmax)

        '''
        self.y_index =  tf.argmax(self.input_y,1,output_type=tf.int32)
        self.indexes = tf.range(0, tf.shape(self.outputs)[0]) * tf.shape(self.outputs)[1] + self.y_index
        self.responsible_outputs = - tf.reduce_mean(tf.log(tf.gather(tf.reshape(self.outputs, [-1]),self.indexes)))
        '''
        # loss 
        self.cross_loss = -tf.reduce_mean( tf.log(tf.reduce_sum( self.input_y  * self.outputs ,axis=1)))
        self.reward =  tf.log(tf.reduce_sum( self.input_y  * self.outputs ,axis=1))

        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())

        self.sum_loss = self.cross_loss +self.gamma2 * self.l2_loss
        self.final_loss = self.sum_loss + 0.2*self.loss0 +  0.1 * self.loss1 + 0.2*self.loss2

        #accuracy
        self.pred = tf.argmax(self.outputs,axis=1)
        self.pred_prob = tf.reduce_max(self.outputs,axis=1)

        self.y_label = tf.argmax(self.input_y,axis=1)
        self.accuracy = tf.reduce_mean(tf.cast( tf.equal(self.pred,self.y_label), 'float'))


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


def train(path_train_word,path_train_pos1,path_train_pos2,path_train_y,path_entity1_category,path_entity2_category,path_father_node,path_relation_to_father,path_pos,path_tokens_to_keep,save_path,gamma1,gamma2):


    print('reading wordembedding')
    wordembedding = np.load('../entity_type/data/vec.npy')
    typeembedding = np.load('../entity_type/data/type_embedding.npy')

    print('reading training data')

    cnn_train_word = np.load(path_train_word)
    cnn_train_pos1 = np.load(path_train_pos1)
    cnn_train_pos2 = np.load(path_train_pos2)
    cnn_train_y    = np.load(path_train_y)
    cnn_entity1_category = np.load(path_entity1_category)
    cnn_entity2_category = np.load(path_entity2_category)
    # cnn_dependency = np.load(path_dependency)[indices]
    cnn_father_node = np.load(path_father_node)
    cnn_relation_to_father = np.load(path_relation_to_father)
    cnn_pos = np.load(path_pos)
    cnn_tokens_to_keep = np.load(path_tokens_to_keep)

    settings = Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(cnn_train_y[0])
    settings.num_steps = len(cnn_train_word) // settings.batch_size
    

    config = tf.ConfigProto(allow_soft_placement=True)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    config.gpu_options.allow_growth = False

 #   config.gpu_options.visible_device_list = ['1']
    #with tf.device('/gpu:0'):
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    with tf.Graph().as_default():
        sess = tf.Session(config=config)
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = CNN(word_embeddings=wordembedding,type_embeddings = typeembedding, setting=settings,gamma1=gamma1,gamma2 = gamma2)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            #saver.restore(sess,save_path=save_path)              
            for epoch in range(1,settings.num_epochs+1):

                bar = tqdm(range(settings.num_steps), desc='epoch {}, loss=0.000000, accuracy=0.000000,sum_loss=0.000000,loss0=0.000000,loss1=0.000000,loss2=0.000000'.format(epoch))
                index_list = [i for i in range(len(cnn_train_y))]
                random.shuffle(index_list)
    #            print(index)
                batch_index = 0
                for _ in bar:
                    sample_list = []
                    #sample_list = index_list[0:10]
                    #print(index_list[0:10])
                    #sample_list = random.sample(range(len(cnn_train_y)),settings.batch_size)
                    #print(str(settings.batch_size * batch_index)+"\t"+str(settings.batch_size * (batch_index+1)))
                    sample_list = index_list[(settings.batch_size * batch_index):(settings.batch_size * (batch_index+1))]
                    batch_index = batch_index + 1
                    
                    batch_train_word = [cnn_train_word[x] for x in sample_list]
                    batch_entity1_category = [cnn_entity1_category[x] for x in sample_list]
                    batch_entity2_category = [cnn_entity2_category[x] for x in sample_list]
                    batch_train_y = [cnn_train_y[x] for x in sample_list]
                    batch_train_pos1 = [cnn_train_pos1[x] for x in sample_list]
                    batch_train_pos2 = [cnn_train_pos2[x] for x in sample_list]
                    # batch_dependency = [cnn_dependency[x] for x in sample_list] 
                    batch_father_node = [cnn_father_node[x] for x in sample_list] 
                    batch_relation_to_father = [cnn_relation_to_father[x] for x in sample_list]
                    batch_tokens_to_keep = [cnn_tokens_to_keep[x] for x in sample_list]
                    batch_pos = [cnn_pos[x] for x in sample_list]                   

                    entity1_category = []
                    entity2_category = []
                    for index in range(len(batch_train_word)):
                        e1 = []
                        e2 = []
                        e1.append(batch_entity1_category[index].tolist().index(1))
                        e2.append(batch_entity2_category[index].tolist().index(1))
                        entity1_category.append(e1)
                        entity2_category.append(e2)

                    feed_dict = {}
                    feed_dict[model.input_word] = batch_train_word
                    feed_dict[model.entity1_category] = entity1_category
                    feed_dict[model.entity2_category] = entity2_category
                    # feed_dict[model.entity1_position] = batch_entity1_position
                    # feed_dict[model.entity2_position] = batch_entity2_position
                    feed_dict[model.input_pos1] = batch_train_pos1
                    feed_dict[model.input_pos2] = batch_train_pos2
                    feed_dict[model.input_y] = batch_train_y
                    feed_dict[model.keep_prob] = settings.keep_prob
               
                    # feed_dict[model.input_dependency] = batch_dependency
                    feed_dict[model.father_node] = batch_father_node
                    feed_dict[model.relation_to_father] = batch_relation_to_father
                    feed_dict[model.tokens_to_keep] = batch_tokens_to_keep
                    feed_dict[model.pos] = batch_pos

                    #print(batch_dependency)
                    _,loss,step,sum_loss,loss0,loss1,loss2=sess.run([model.train_op, model.final_loss, model.accuracy,model.sum_loss,model.loss0,model.loss1,model.loss2],feed_dict=feed_dict)
                    bar.set_description('epoch {} loss={:.6f} step={:.6f} sum_loss={:.6f},loss0={:.6f},loss1={:.6f},loss2={:.6f}'.format(epoch, loss, step,sum_loss,loss0,loss1,loss2))
                save_path =  '../entity_type/entityType_connect/model_' + str(gamma1) +'_'+str(gamma2)+'epoch'+str(epoch)+ '_new_.ckpt'
                saver.save(sess, save_path=save_path)

                modeltest.produce_pred_data(save_path=save_path,output_path = '../entity_type/result/origin_pred_entitypair.pkl')
                result = modeltest.P_N(label_path = '../entity_type/data/label_entitypair.pkl',pred_path ='../entity_type/result/origin_pred_entitypair.pkl')
                print('origin_cnn_P@100,200,300:', result)
                if result[0]>=0.95 and result[1] >= 90:
                    save_path = '../entity_type/bestmodel/model_' + str(gamma1) +'_'+str(gamma2)+'epoch'+str(epoch)+ '_best_.ckpt'
                    saver.save(sess, save_path=save_path)



class interaction():

    def __init__(self,sess,save_path ='../entity_type/model/model.ckpt3'):

        self.settings = Settings()
        wordembedding = np.load('../entity_type/data/vec.npy').astype('float32')
        typeembedding = np.load('../entity_type/data/type_embedding.npy').astype('float32')
        self.settings.is_training = False
        self.sess = sess
        with tf.variable_scope("model"):
            self.model = CNN(word_embeddings=wordembedding,type_embeddings = typeembedding, setting=self.settings,gamma1 = 0.1,gamma2 = 0.2)

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess,save_path)

        self.train_word = np.load('../entity_type/data/train_word.npy')
        self.train_pos1 = np.load('../entity_type/data/train_pos1.npy')
        self.train_pos2 = np.load('../entity_type/data/train_pos2.npy')
        self.y_train = np.load('../entity_type/data/train_y.npy')
        # entity_type/cnndata/cnn_train_entity1.npy

    def test(self,batch_test_word,batch_test_pos1,batch_test_pos2,path_entity1_category,path_entity2_category):
        feed_dict = {}
        feed_dict[self.model.input_word] = batch_test_word
        feed_dict[self.model.input_pos1] = batch_test_pos1
        feed_dict[self.model.input_pos2] = batch_test_pos2
        feed_dict[self.model.keep_prob] = 1
        entity1_category = []
        entity2_category = []
        for index in range(len(batch_test_word)):
            e1 = []
            e2 = []
            e1.append(path_entity1_category[index].tolist().index(1))
            e2.append(path_entity2_category[index].tolist().index(1))
            entity1_category.append(e1)
            entity2_category.append(e2)
        feed_dict[self.model.entity1_category] = entity1_category
        feed_dict[self.model.entity2_category] = entity2_category
        relation,prob = self.sess.run([self.model.pred,self.model.pred_prob],feed_dict = feed_dict)

        return (relation,prob)


if __name__ == '__main__':

    gamma1 = float(sys.argv[1])
    gamma2 = float(sys.argv[2])
    model_name = '../entity_type/model/cnnmodel_union_transformer_with_cnn_with_entity_desc_' + str(gamma1) +'_'+str(gamma2)+ 'new_.ckpt'
    print(model_name)

    # train model
    print ('train model')
    train('../entity_type/cnndata/cnn_train_word.npy','../entity_type/cnndata/cnn_train_pos1.npy', '../entity_type/cnndata/cnn_train_pos2.npy', '../entity_type/cnndata/cnn_train_y.npy', \
        '../entity_type/cnndata/cnn_train_entity1.npy','../entity_type/cnndata/cnn_train_entity2.npy','../entity_type/cnndata/cnn_train_parent.npy', '../entity_type/cnndata/cnn_train_relation_to_parent.npy', '../entity_type/cnndata/cnn_train_part_of_word.npy', '../entity_type/cnndata/cnn_train_mask.npy',model_name,gamma1,gamma2)

    # #get embedding
    # print('get_embedding')
    # get_sentence_embedding('cnndata/cnn_train_word.npy','cnndata/cnn_train_y.npy', 'model/origin_cnn_model.ckpt')







