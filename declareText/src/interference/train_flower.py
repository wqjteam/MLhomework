import os

import numpy as np
import tensorflow as tf

from src.dataset.Preprocess_data import generate_batch, process_declare
from src.models.model import rnn_model

tf.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')

# set this to 'main.py' relative path
tf.flags.DEFINE_string('checkpoint', os.path.abspath('../checkpoints/model/declare/'),
                       'checkpoints and model save path.')
# tf.flags.DEFINE_string('file_path', os.path.abspath('../dataset/data/poems.txt'), 'file name of poems.')
tf.flags.DEFINE_string('file_path', os.path.abspath('../dataset/data/wordtotxt/流深数据可视化分析平台软件V1.0_使用说明书.txt'),
                       'file name of poems.')
tf.flags.DEFINE_string('events_file_path', os.path.abspath('../checkpoints/tmp/summary/test/'), 'file name of poems.')

tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')

tf.flags.DEFINE_integer('epochs', 5000, 'train how many epochs.')

FLAGS = tf.flags.FLAGS

start_token = 'G'
end_token = 'E'


def run_training():
    if not os.path.exists(os.path.dirname(FLAGS.checkpoint)):
        os.mkdir(os.path.dirname(FLAGS.checkpoint))
    if not os.path.exists(FLAGS.checkpoint):
        os.mkdir(FLAGS.checkpoint)

    data_vector, word_to_int, vocabularies = process_declare(FLAGS.file_path)
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, data_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=3, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # 收集变量 单个数字值收集
    print(end_points['loss'], end_points['total_loss'])
    # tf.summary.scalar("loss", end_points['loss'])
    tf.summary.scalar("total_loss", end_points['total_loss'])
    tf.summary.scalar("acc", 1)
    # 定义一个合并变量de op
    merged = tf.summary.merge_all()
    # 高纬度变量收集
    # tf.summary.histogram("weightes", weight)
    # tf.summary.histogram("biases", bias)

    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init_op)

        # 建立events文件，然后写入
        filewriter = tf.summary.FileWriter(FLAGS.events_file_path, graph=sess.graph)
        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint + '\\')
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('[INFO] start training...')
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                n_chunk = len(data_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('[INFO] Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
                    # 写入每步训练的值
                    summary = sess.run(merged,
                                       feed_dict={input_data: batches_inputs[0], output_targets: batches_outputs[0]})
                    filewriter.add_summary(summary, batch)
                # 将数据写入tensorboard中

                if epoch % 6 == 0:
                    saver.save(sess, FLAGS.checkpoint + "/", global_step=epoch)
                    # saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(epoch))


def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def gen_declare(begin_word):
    batch_size = 1
    print('[INFO] loading corpus from %s' % FLAGS.file_path)
    poems_vector, word_int_map, vocabularies = process_declare(FLAGS.file_path)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=3, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        # checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
        saver.restore(sess, checkpoint)
        # saver.restore(sess, './model/-24')

        # 吧输入的文字转为数字
        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, vocabularies)
        poem = ''
        while word != end_token:
            print('runing')
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_int_map[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)
        # word = words[np.argmax(probs_)]
        return poem


def pretty_print_poem(poem):
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


def main(is_train):
    if is_train:
        print('[INFO] train declare_txt...')
        run_training()
    else:
        print('[INFO] write declare_txt...')

        begin_word = input('输入起始字:')
        # begin_word = '我'
        poem2 = gen_declare(begin_word)
        pretty_print_poem(poem2)


if __name__ == '__main__':
    main(True)  # traing
    # main(False)
