#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
import argparse
import codecs
import sys


def main():
  parser = argparse.ArgumentParser('convert conll file to input format')
  parser.add_argument("--input", help="path to input file", default="")
  parser.add_argument("--output", help="path to output file", default="")
  args = parser.parse_args()

  if args.input == "":
    print("input file needed", file=sys.stderr)
    sys.exit(1)
  if args.output == "":
    print("output file needed", file=sys.stderr)
    sys.exit(1)
  if args.input == args.output:
    print("input and output cannot be the same", file=sys.stderr)
    sys.exit(1)

  output_file = codecs.open(args.output, "w", encoding='utf-8')

  try:
    sent = []
    for line in codecs.open(args.input, "r", encoding='utf-8'):
      line = line.strip()
      if len(line) == 0:
        print(u' '.join(u'{0}_{1}'.format(word, postag) for word, postag in sent), file=output_file)
        sent = []
      else:
        fields = line.split()
        word, postag = fields[1], fields[3]
        sent.append((word, postag))
    if len(sent) > 0:
      print(u' '.join(u'{0}_{1}'.format(word, postag) for word, postag in sent), file=output_file)
  finally:
    output_file.close()


if __name__ == '__main__':
  main()
