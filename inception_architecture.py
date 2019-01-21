import ops as op
import tensorflow as tf


def placeholders(img_size,img_channel,label_cnt):
	with tf.name_scope('input'):
		X=tf.placeholder(shape=[None,img_size,img_size,img_channel],dtype=tf.float32,name='image')
		y=tf.placeholder(shape=[None,label_cnt],dtype=tf.float32,name='target')

	with tf.name_scope('hparams'):
		learning_rate=tf.placeholder(shape=None,dtype=tf.float32,name='learning_rate')
		dropout_keep_prob=tf.placeholder(shape=None,dtype=tf.float32,name='keep_prob')

	training=tf.placeholder(shape=None,dtype=tf.bool,name='is_train')
	return X,y,learning_rate,dropout_keep_prob,training




def inception_block(X,conv3x3reduce_,conv1x1_,conv3x3_,conv5x5reduce_,conv5x5_,pool_proj_,name,training):
	with tf.name_scope(name):
		conv1x1=op.conv(X,filter_size=1,stride_size=1,padding='SAME',out_channels=conv1x1_,a=None)
		conv3x3reduce=op.conv(X,filter_size=1,stride_size=1,padding='SAME',out_channels=conv3x3reduce_,a=tf.nn.relu)
		conv3x3=op.conv(X,filter_size=3,stride_size=1,padding='SAME',out_channels=conv3x3_,a=None)
		conv5x5reduce=op.conv(X,filter_size=1,stride_size=1,padding='SAME',out_channels=conv5x5reduce_,a=tf.nn.relu)
		conv5x5=op.conv(X,filter_size=5,stride_size=1,padding='SAME',out_channels=conv5x5_,a=None)
		pool=op.maxpool(X,filter_size=3,stride_size=2,padding='VALID')
		pool_proj=op.conv(X,filter_size=1,stride_size=1,padding='SAME',out_channels=pool_proj_,a=None)
		
		conv1x1=op.batch_norm(conv1x1,training=training,name=name+'_conv1x1_batchnorm')
		conv3x3=op.batch_norm(conv3x3,training=training,name=name+'_conv3x3_batchnorm')
		conv5x5=op.batch_norm(conv5x5,training=training,name=name+'_conv5x5_batchnorm')
		pool_prol=op.batch_norm(pool_proj,training=training,name=name+'_pool_batchnorm')

		conv1x1=tf.nn.relu(conv1x1)
		conv3x3=tf.nn.relu(conv3x3)
		conv5x5=tf.nn.relu(conv5x5)
		pool_prol=tf.nn.relu(pool_proj)
		
		out=tf.concat([conv1x1,conv3x3,conv5x5,pool_proj],axis=3,name='output_block')
	return out


def inception3a(X,training):
	return inception_block(X,conv1x1_=64,conv3x3reduce_=96,conv3x3_=128,conv5x5reduce_=16,conv5x5_=32,pool_proj_=32,name='inception3a',training=training)
def inception3b(X,training):
	return inception_block(X,conv1x1_=128,conv3x3reduce_=128,conv3x3_=192,conv5x5reduce_=32,conv5x5_=96,pool_proj_=64,name='inception3b',training=training)
def inception4a(X,training):
	return inception_block(X,conv1x1_=192,conv3x3reduce_=96,conv3x3_=208,conv5x5reduce_=16,conv5x5_=48,pool_proj_=64,name='inception4a',training=training)
def inception4b(X,training):
	return inception_block(X,conv1x1_=160,conv3x3reduce_=112,conv3x3_=224,conv5x5reduce_=24,conv5x5_=64,pool_proj_=64,name='inception4b',training=training)
def inception4c(X,training):
	return inception_block(X,conv1x1_=128,conv3x3reduce_=128,conv3x3_=256,conv5x5reduce_=24,conv5x5_=64,pool_proj_=64,name='inception4c',training=training)
def inception4d(X,training):
	return inception_block(X,conv1x1_=112,conv3x3reduce_=144,conv3x3_=288,conv5x5reduce_=32,conv5x5_=64,pool_proj_=64,name='inception4d',training=training)
def inception4e(X,training):
	return inception_block(X,conv1x1_=256,conv3x3reduce_=160,conv3x3_=320,conv5x5reduce_=32,conv5x5_=128,pool_proj_=128,name='inception4e',training=training)
def inception5a(X,training):
	return inception_block(X,conv1x1_=256,conv3x3reduce_=160,conv3x3_=320,conv5x5reduce_=32,conv5x5_=128,pool_proj_=128,name='inception5a',training=training)
def inception5b(X,training):
	return inception_block(X,conv1x1_=384,conv3x3reduce_=192,conv3x3_=384,conv5x5reduce_=48,conv5x5_=128,pool_proj_=128,name='inception5b',training=training)


def auxillary_logits(X,label_cnt,name):
	with tf.name_scope(name):
		auxillary_logits1=op.avgpool(X,filter_size=5,stride_size=3,padding='VALID')
		auxillary_logits1=op.conv(auxillary_logits1,out_channels=1024,filter_size=1,stride_size=1,padding='SAME',a=tf.nn.relu) # not sure as per the no. of out_channels in this conv
		auxillary_logits1=op.fc(auxillary_logits1,output_size=2048,a=tf.nn.relu) # not sure as per the output size in this fc layer
		auxillary_logits1=op.fc(auxillary_logits1,output_size=label_cnt,a=None)
	return auxillary_logits1


def network(X,training,label_cnt,dropout_keep_prob):
	with tf.name_scope('pre_inception'):
		with tf.name_scope('conv1layer'):
			X=op.conv(X,filter_size=7,stride_size=2,padding='VALID',out_channels=64,a=tf.nn.relu)
			X=tf.pad(X,[[0,0],[1,1],[1,1],[0,0]])
			X=op.maxpool(X,filter_size=3,stride_size=2,padding='VALID')
			X=op.lrn(X)
		with tf.name_scope('conv2layer'):
			X=op.conv(X,filter_size=3,stride_size=1,padding='SAME',out_channels=192,a=tf.nn.relu)
			X=op.maxpool(X,filter_size=3,stride_size=2,padding='VALID')
			X=op.lrn(X)
	with tf.name_scope('inception_blocks'):
		X=inception3a(X,training)
		X=inception3b(X,training)
		X=inception4a(X,training)
		logits1=auxillary_logits(X,label_cnt,name='auxillary_layer1')
		X=inception4b(X,training)
		X=inception4c(X,training)
		X=inception4d(X,training)
		logits2=auxillary_logits(X,label_cnt,name='auxillary_layer2')
		X=inception4e(X,training)
		X=inception5a(X,training)
		X=inception5b(X,training)
	with tf.name_scope('post_inception'):
		X=op.avgpool(X,filter_size=7,stride_size=1,padding='VALID')
		X=tf.nn.dropout(X,dropout_keep_prob)
	with tf.name_scope('fc1layer'):
		final_logits=op.fc(X,output_size=label_cnt,a=None)
	with tf.name_scope('Softmax'):
		out_probs=tf.nn.softmax(logits=X,axis=-1,name='softmax_op')

	return logits1,logits2,final_logits,out_probs



def loss(logits1,logits2,final_logits,labels,importance=[0.2,0.3,0.5],type_='auxillary1'):
	with tf.name_scope('loss1'):
		loss_1=tf.multiply(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits1,labels=labels)),importance[0])
	with tf.name_scope('loss2'):
		loss_2=tf.multiply(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2,labels=labels)),importance[1])
	with tf.name_scope('loss3'):
		loss_3=tf.multiply(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_logits,labels=labels)),importance[2])
	final_loss=add_loss(loss_1,loss_2,loss_3)
	tf.summary.scalar('auxillary_loss1',loss_1)
	tf.summary.scalar('auxillary_loss2',loss_2)
	tf.summary.scalar('final_loss',final_loss)
	return final_loss

def add_loss(loss1,loss2,loss3):
	return tf.add_n([loss1,loss2,loss3],name='sum_loss')

def optimizer(loss,learning_rate):
	with tf.name_scope('optimizer'):
		opt=tf.train.AdamOptimizer(learning_rate)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op=opt.minimize(loss)
	return train_op

def accuracy(logits,labels):
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	return accuracy