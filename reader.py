from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf
import re

from glob import glob

def _read_words(filename):
  #return a list where enlines is replaced with <eos> tags
  file = open(filename, "r")
  s = file.read().split("\n")
  return s


def _get_stuff(filelements):
  notes1 = []
  notes2 = []
  notes3 = []
  notes4 = []

  for i in filelements:
    # print (str(i[0]))
    if(i == ""):
      continue
    if str(i[0]) == "!" or str(i[0]) == "*" or str(i[0]) =="=":
      continue

    strings = i.split("\t")
        # print(strings)
    part1 = strings[0]
    part2 = strings[1]
    part3 = strings[2]
    part4 = strings[3]

    notes1.append(_string_to_idx(part1))
    notes2.append(_string_to_idx(part2))
    notes3.append(_string_to_idx(part3))
    notes4.append(_string_to_idx(part4))

  notes1.append(88)
  notes2.append(88)
  notes3.append(88)
  notes4.append(88)

  allNotes = notes1 + notes2 + notes3 + notes4
  return allNotes


musicNotes = ["a","a\#", "b", "c", "c#"]

def _string_to_idx(string):

  temp = string


  if string == "." :
    return 89

  string = string.replace(";","").replace("J","").replace("L","").replace(".","").replace("X","")

  if string[0].isdigit() :
    string=string[1:]

  if string[0].isdigit() :
    string=string[1:]
  if string[0].isdigit() :
    string=string[1:]

  same = string.split('\t')
  string = re.sub('\d', '', string)
  count = 0;
  if string[len(string)-1]=="-":
    count = count - 1
    string=string[:1]
    if string[len(string)-1]=="-":
      count = count - 1
      #print (string)
      if string[len(string)-1]=="-":
        count = count - 1
        string = string[:1]

  if string[len(string)-1]=="#":
    count = count + 1
    string = string[:1]

    if string[len(string)-1]=="#":
      count = count + 1
      string = string[:1]
      if string[len(string)-1]=="#":
        count = count + 1
        string = string[:1]

  tot = string
  temp = re.sub(r'\W+', '', string)

  s = len(temp)
  
  octave = 0
  if string[0].istitle():
    octave = 36-(s*12)
  else :
    octave = (s-1)*12+36

  char = string[0].lower()
  ints = ord(char) - 96

  if octave+count+ints < 0 or octave+count+ints>90:
    print("HAHAHAHAHHDFJ FASKLDJFCED UR FUCKEDKASKDJFKASDJFKJ")

  return octave+count+ints




def _file_to_word_ids(filename, word_to_id):
  #removes the not punc
  
  return real_data


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "news.en-00001-of-00100")
  valid_path = os.path.join(data_path, "news.en-00002-of-00100")
  test_path = os.path.join(data_path, "news.en-00003-of-00100")
  
  word_to_id = _build_vocab(train_path)

  sos = "<sos>"
  train_data = _read_words(train_path)
  train_data = [sos] + train_data
  valid_data = _read_words(valid_path)
  valid_data = [sos] + valid_data
  test_data = _read_words(test_path )
  test_data = [sos] + test_data
  vocabulary = len(word_to_id)
  train_data.pop()
  valid_data.pop()
  test_data.pop()
  #print(vocabulary)
  return train_data, valid_data, test_data, word_to_id

def ptb_raw_data_test(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """
  s = "/home/supersteve/371chorales"
  data = glob(os.path.join(s,"*.krn"))
  raw_data = []
  for i in data:
    raw_data = raw_data +(_get_stuff(_read_words(i)))
  
  
  #print(vocabulary)
  return raw_data

def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  #raw_data = np.asarray(raw_data)

  print(type(raw_data))


  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    print(np.array(raw_data))
    raw_data = tf.convert_to_tensor(np.array((raw_data)), name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])

    y.set_shape([batch_size, num_steps])
    return x, y
