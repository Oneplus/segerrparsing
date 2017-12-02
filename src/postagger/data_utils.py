#!/usr/bin/env python
# coding=utf-8

# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--input", help="path to input file", default="")
# parser.add_argument("--output", help="path to output file", default="")
# args = parser.parse_args()
#
# if __name__ == '__main__':
#   if (args.input == ""):
#     print
#     "input file needed"
#     exit(0)
#   if (args.output == ""):
#     print
#     "output file needed"
#     exit(0)
#   if (args.input == args.output):
#     print
#     "input and output cannot be the same"
#     exit(0)
#
#   input_file = open(args.input, "r")
#   output_file = open(args.output, "w")
#
#   try:
#     input_lines = input_file.readlines()
#     sent = []
#     feature = []
#     for line in input_lines:
#       if (line == "\n"):
#         data = []
#         labels = []
#         for x in sent:
#           data.append(x.split()[0])
#           labels.append(x.split()[2])
#         output_file.write(' '.join(data) + '\t' + ' '.join(labels) + '\n')
#         sent = []
#       else:
#         sent.append(line[:-1])
#   finally:
#     input_file.close()
#     output_file.close()
#


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="path to input file", default="../../data/CTB5.1-test.gp.conll")
parser.add_argument("--output", help="path to output file", default="../../data/test_pos.txt")
args = parser.parse_args()

if __name__ == '__main__':
  if (args.input == ""):
    print
    "input file needed"
    exit(0)
  if (args.output == ""):
    print
    "output file needed"
    exit(0)
  if (args.input == args.output):
    print
    "input and output cannot be the same"
    exit(0)

  input_file = open(args.input, "r")
  output_file = open(args.output, "w")

  try:
    str_res = ""
    result = ""
    input_lines = input_file.read().strip().split('\n\n')
    for index_conll, value_conll in enumerate(input_lines):
      value_conll = value_conll.strip().split('\n')
      for index_word, value_word in enumerate(value_conll):
        value_word = value_word.strip().split()  # convert to list type
        str_res += value_word[1] + "_" + value_word[3] + ' '
      str_res += '\n'
    result+= str_res
    output_file.write(result)
  finally:
    input_file.close()
    output_file.close()

