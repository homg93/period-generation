from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from collections import OrderedDict
import time
import matplotlib.pyplot as plt

tf.set_random_seed(777)    # reproducibility

word_set = []

def read_sentence(readfile):
    #srt_merge_result2.txt를 불러오는 함수
    s = []
    srt = []    
    with open(readfile, 'r',encoding = "utf-8") as f:
        
        for line in f:
            inputs = line.split("\n")
            s = inputs[0].strip().split()
            
            srt.append(s)
    return srt

            # print(s_len-1)

# srt = read_sentence("read_sentence2.txt")
# test_srt = read_sentence("read_sentence2.txt")

srt = read_sentence("read_sentence.txt")
test_srt = read_sentence("test_read_sentence.txt")

# print all of word
# for i in range(len(srt)):
#     for j in range(len(srt[i])):
#         print("{} : {}".format(j,srt[i][j]))

# for i in range(len(srt)):
#     read_all_of_word_cnt = len(srt[i])


word_set = list(set())
words = []
wordvec = []


def read_vocab_dic(readfile):
    #load to vocab.txt 
    word = []
    
    with open(readfile, 'r',encoding = "utf-8") as f:
        
        for line in f:
            if line == '\n':
                continue

            s = line.strip().split()
            word = np.array(s)

            inputs = line.split("\n")
            s = inputs[0]
            words.append(word[0])
            wordvec_np = np.array(word[1:],dtype = np.float32)
            # wordvec_np = word[1:].astype(float)
            wordvec.append(wordvec_np)
            # print(wordvec)
            # word_dic = {w: wordvec[i] for i, w in enumerate(words)}
            word_dic = {w: i for i, w in enumerate(words)}
            word_dic2 = {i: w for i, w in enumerate(words)}

    return word_dic, word_dic2

word_dic, word_dic2 = read_vocab_dic("w2v_100_srt_merge_result.txt")
print(len(words))
wordvec = np.array(wordvec)

# print(word_dic.values()) #숫자 
# for i in range(len(wordvec)):
#     print(i)
#     print(words[i])
#     print(wordvec[i])

data_dim = len(word_dic)
hidden_size = 5
word_length = 5    # 5개씩 본다
learning_rate = 0.01


all_of_dataX = []
def build_data(srt, word_length):
    dataX = []
    dataY_before_set = []
    dataY = set()
    dataY_dict = {'N' : 0, 'A' : 1, 'B' : 2, 'C' : 3, 'D' : 4, 'E' : 5, 'F' : 6,}

    for j in range(0,len(srt)):
        for i in range(0, len(srt[j]),5):
            if len(srt[j]) - i >5:
                x_str = srt[j][i:i + word_length]
                y_str = [0,0,0,0,0]

                have_period = 0
                for k in range(len(x_str)):
                    # print(x_str[k])
                    if x_str[k] == '.':
                        y_str[k] = 1
                        have_period = k+1
                    if x_str[k] not in word_dic:
                        x_str[k] = 'UNK'
                # y_str = srt[j][i + 1: i + word_length + 1]
                if have_period >= 1:
                    # print(j, i, x_str, '->', y_str)
                    for nx in range(have_period-1,5):
                        x_str[nx] = srt[j][i+nx+1]
                        if x_str[nx] not in word_dic:
                            x_str[nx] = 'UNK'
                    print(j, i, x_str, '->', y_str)


            x = [word_dic[c] for c in x_str]    # x str to index

            if y_str[4] == 1:
                inputY = 'E'
                dataX.append(x)
                all_of_dataX.append(x)
                dataY_before_set.append([0,0,0,0,1])
            elif y_str[3] == 1:
                inputY = 'D'
                dataX.append(x)
                all_of_dataX.append(x)
                dataY_before_set.append([0,0,0,1,0])
            elif y_str[2] == 1:
                inputY = 'C'
                dataX.append(x)
                all_of_dataX.append(x)
                dataY_before_set.append([0,0,1,0,0])
            elif y_str[1] == 1:
                inputY = 'B'
                dataX.append(x)
                all_of_dataX.append(x)
                dataY_before_set.append([0,1,0,0,0])
            elif y_str[0] == 1:
                inputY = 'A'
                dataX.append(x)
                all_of_dataX.append(x)
                dataY_before_set.append([1,0,0,0,0])
            # else:
            #     inputY = 'N'
            #     dataY_before_set.append([0,0,0,0,0])
            
            # dataX.append(x)#vec 100 or index
            # dataX.append(x_str)#word
            
            # inputY_num = [dataY_dict[c] for c in inputY]
            # dataY_before_set.append(y_str)

            # dataY_before_set.append([0,0,0,0,0,0,0])
            # dataY.update(inputY_num)
    return dataX, dataY_before_set, dataY

dataX, dataY_before_set, dataY = build_data(srt, word_length)
test_dataX, test_dataY_before_set, test_dataY = build_data(test_srt, word_length)

dataY = list(dataY)
dataY = [[1,0,0,0,0],[0,0,0,0,1],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]

dataX = np.array(dataX)
dataY = np.array(dataY)
dataY_before_set = np.array(dataY_before_set)

# for i in range(len(dataX)):
#     print("dataX: {}\t dataY: {}".format(dataX[i],dataY_before_set[i]))

# for i in range(len(test_dataX)):
#     print("test_dataX: {}\t test_dataY_before_set: {}".format(test_dataX[i],test_dataY_before_set[i]))

print("!!!!!!!!!!!!!! dataX,Y")
print(len(dataX))
print(len(dataY_before_set))
print("!!!!!!!!!!!!!! test data X,Y")
print(len(test_dataX))
print(len(test_dataY_before_set))

# batch_size = 20
batch_size = len(dataX)
num_classes = len(dataY)

X = tf.placeholder(tf.int32, [None, word_length])
Y = tf.placeholder(tf.int32, [None, len(dataY)])

# One-hot encoding
X_one_hot = tf.one_hot(X, len(dataX)) # choose 1 having 2
Y_one_hot = tf.one_hot(Y, len(dataY)) # choose 1 having 2
Y_one_hot = tf.reshape(Y_one_hot, [-1, num_classes])
# print("X_one_hot")
print(X_one_hot)    # check out the shape
print("Y_one_hot: {}".format(Y_one_hot))    # check out the shape
print(len(dataY))
print(dataY)




# Make a lstm cell with hidden_size (each unit output vector size)
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)
print("outputs.shape: {}".format(outputs.shape))

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
# outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=tf.nn.relu)


# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, word_length, num_classes])
print("outputs.shape: {}".format(outputs.shape))

# All weights are 1 (equal weights)
weights = tf.ones([batch_size, word_length])
print("weights.shape: {}".format(weights.shape))

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

correct_pred = tf.equal(tf.argmax(outputs, 1), tf.argmax(dataY, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# tf_embedding = tf.constant(wordvec, dtype=tf.float32)
# input_str = "expose positives"
# word_to_idx = OrderedDict({w:words.index(w) for w in dataX if w in words})
# train_inputs = tf.placeholder(tf.float32, shape=[batch_size])



###########################################
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# # word_embedding = tf.nn.embedding_lookup(train_inputs, list(word_to_idx.values()))
# train_loss_list = []
# val_loss_list = []

# for i in range(500):
#     _, l, results = sess.run([train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY_before_set})
#     l2, results2 = sess.run([mean_loss, outputs], feed_dict={X: test_dataX, Y: test_dataY_before_set})
#     if i % 10 == 0:
#         print("epochs {}".format(i))
#         print("cost : {}".format(l))

#         print("cv_cost : {}".format(l2))
#         train_loss_list.append(l)
#         val_loss_list.append(l2)
#         # print("results : {}".format(results))
#     for j, result in enumerate(results):
#         index = np.argmax(result, axis=1)
#         # print(index)
#         # print(i, j, ''.join([dataY_dict[t] for t in index]), l)

# # Let's print the last char of each result to check it works
# results_list= []
# result_cnt = 0

# test_results = sess.run(outputs, feed_dict={X: test_dataX})
# for j, result in enumerate(test_results):
#     index = np.argmax(result, axis=1)
#     if j is 0:    # print all for the first result to make a sentence
#         # print(''.join([word_dic[t] for t in index]), end='')
#         results_list.append(index)
#     else:
#         # print(word_dic[index[-1]], end='')
#         results_list.append(index)

# # dataY_before_set = list(dataY_before_set)
# # results_list = list(results_list)


# # evaluate
# print("len(results_list): {}".format(len(results_list)))
# print(results_list[1])
# print("len(test_dataY_before_set): {}".format(len(test_dataY_before_set)))
# print(test_dataY_before_set[1])

# results_list = np.array(results_list)
# test_dataY_before_set = np.array(test_dataY_before_set)
# print(type(results_list))
# print(type(test_dataY_before_set))

# #write result
# t3 = time.localtime()

# result_writefile = "result"+str(t3.tm_mon)+str(t3.tm_mday)+"_"+str(t3.tm_hour)+ str(t3.tm_min)+ str(t3.tm_sec)+".txt"
# result_writefile2 = "result_cor"+str(t3.tm_mon)+str(t3.tm_mday)+"_"+str(t3.tm_hour)+ str(t3.tm_min)+ str(t3.tm_sec)+".txt"
# err_list = []
# cor_list = []
# cor_word = []
# err_word = []
# # outfile = open(result_writefile,"w", encoding = "utf-8")
# # outfile_cor = open(result_writefile2,"w", encoding = "utf-8")
# test_len3 = int(len(test_dataY_before_set)/3)
# for i in range(test_len3):
#     cnt = 0
#     for j in range(5):
#         if test_dataY_before_set[i][j] ==  results_list[i][j]:
#             cnt = cnt + 1

#     if cnt == 5:
#         cor_list.append(i)
#         result_cnt = result_cnt + 1
#         for j in range(5):
#             if results_list[i][j] == 1:
#                 cor_word.append(word_dic2[test_dataX[i][j]])
#     else:
#         for j in range(5):
#             if results_list[i][j] == 1:
#                 err_word.append(word_dic2[test_dataX[i][j]])
#         err_list.append(i)
#         print("정답[{}]: {}".format(i,test_dataY_before_set[i]))
#         print("예측[{}]: {}".format(i,results_list[i]))
#         print("test_word_list[{}]: {}".format(i,test_dataX[i]))
# cor_word = set(cor_word)
# err_word = set(err_word)

# def write_txt(writefile, listname):
#     outfile = open(writefile,"w", encoding = "utf-8")
#     outfile.write("cost: {}\n".format(train_loss_list[len(train_loss_list)-1]))

    
#     outfile.write("test len: {}\n".format(test_len3))
#     outfile.write("correct result_cnt: {}\n".format(result_cnt))
#     outfile.write("wrong result_cnt: {}\n".format(test_len3 - result_cnt))
#     outfile.write("result_cnt / len: {}\n\n".format(result_cnt / test_len3))
#     outfile.write("==================================\n")
#     outfile.write("cor_word_set: {}\n".format(cor_word))
#     outfile.write("==================================\n")

#     outfile.write("err_word_set: {}\n".format(err_word))
#     outfile.write("==================================\n")
    

#     for i in range(len(listname)):
#         index = listname[i]

#         outfile.write("정답[{}]: {}\n".format(index,test_dataY_before_set[index]))
#         outfile.write("예측[{}]: {}\n".format(index,results_list[index]))
#         x = [word_dic2[c] for c in test_dataX[index]]
#         outfile.write("test_word_list[{}]: {}\n\n".format(index,x))

#     outfile.close()

# write_txt(result_writefile2, cor_list)
# write_txt(result_writefile, err_list)

# print("test len: {}".format(test_len3))
# print("correct result_cnt: {}".format(result_cnt))
# print("result_cnt / len: {}".format(result_cnt / test_len3))

# # print("Testing Accuracy:", \
# #         sess.run(accuracy, feed_dict={X: dataX, Y: dataY_before_set}))
# # print("Testing Accuracy:", \
# #         sess.run(accuracy, feed_dict={X: test_dataX, Y: test_dataY_before_set}))

# x_axis = []
# for i in range(0,50):
#     x_axis.append(i)

# plt.plot(x_axis, train_loss_list, 'b')
# plt.plot(x_axis, val_loss_list, 'g')
# plt.xlabel('Num of epochs')
# plt.ylabel('Cost')
# plt.show()