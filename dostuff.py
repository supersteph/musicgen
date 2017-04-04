from glob import glob
import os
import re

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
    print(strings)
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

  return notes1+notes2+notes3+notes4


musicNotes = {"c":0,"d":2,"e":4,"f":5,"g":7,"a":9,"b":11}

def _string_to_idx(string):

  temp = string


  if string == ".":
    return 89

  string = string.replace(";","").replace("J","").replace("L","").replace(".","").replace("X","").replace("[","").replace("]","")

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
  if char=="r":
    return 89
  ints = musicNotes[char]


  return octave+count+ints

s = "/home/supersteve/371chorales"
data = glob(os.path.join(s,"*.krn"))
raw_data = []

file = open("placeholder.txt", "w")
for i in data:
  a = _get_stuff(_read_words(i))
  
  for s in a:
    file.write(str(s)+" ")

file.close()