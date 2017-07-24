import tensorflow as tf
from random import sample
import sys
from ConvolutionNN import next_y_data,next_x_data,dev_model
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

def randomforest_classifier(x_train,y_train,dev_x,dev_y):
    """RandomForest classification model

            Parameters:
            -----------
            data : training and dev data with their response
            response: accuracy score on dev data

            Returns:
            --------
            accuracy score
            """

    for i in range(10,100):
        model = RandomForestClassifier(n_estimators=i, min_samples_leaf=10, random_state=1)
        model.fit(x_train, y_train)
        accuracy = model.score(dev_x, dev_y)

    return accuracy


def main():

    error_out_file=open('./out/LEARNING_RATE_'+str(sys.argv[1]+'_'+'TRAINING_EPOCHS_'+sys.argv[2]+'_'+'_'+'BATCH_SIZE_'+sys.argv[3])+'_logistic.txt',mode='a')
    learning_rate=float(sys.argv[1])
    epoch = int(sys.argv[2])
    batch_size = int(sys.argv[3])

    n_input=453 # input size
    n_classes=12 # number of classes


    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    w1 = tf.Variable(tf.random_normal([n_input, n_classes]))
    b1 = tf.Variable(tf.random_normal([n_classes]))

    logit = tf.matmul(x, w1) + b1

    prediction = tf.nn.softmax(logit)

    # Output layer with linear activation
    out = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logit)

    # Define loss and optimizer
    cost = tf.reduce_mean(out)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


    correct= tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy= tf.reduce_mean(tf.cast(correct, tf.float32))


    # Y_Train_Data file
    with open('./data/train_data_train.p','rb') as fp:
        train_data=pickle.load(fp)

    with open('./data/train_data_dev.p','rb') as fp:
        train_dev=pickle.load(fp)



    no_of_batches = int(len(train_data)/batch_size)


    #init_op = tf.initialize_all_variables()
    init_op=tf.global_variables_initializer()

    # Create a saver object
    saver = tf.train.Saver()

    max_length=453

    with tf.Session() as sess:

        sess.run(init_op)

        # Training cycle
        for i in range(epoch):

            prev_index = 0

            # # shuffle train data after each epoch
            np.random.shuffle(train_data)

            for j in range(no_of_batches):
                x_train=next_x_data(prev_index,batch_size,train_data,sess,max_length)
                y_train=next_y_data(train_data,prev_index,batch_size,sess)

                prev_index+=batch_size

                sess.run(optimizer,{x: x_train, y: y_train})

            print("Epoch - ",str(i))
            train_dev=sample(list(train_dev),len(train_dev))

            # compute accuracy of model on dev data after each epoch
            dev_model(sess,train_dev,batch_size,max_length,accuracy,prediction,x,y,error_out_file)

         #Now, save the graph
        saver.save(sess, sys.argv[6]+os.sep+str(sys.argv[1]+sys.argv[2]+sys.argv[3]+'convolution_network'))
        print("Optimization Finished!")

    sess.close()



if __name__ == '__main__':
    main()

