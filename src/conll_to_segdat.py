# coding: utf-8
# convert the conll format to the format used as the segmentor input
from __future__ import print_function
import argparse
import codecs


def main():
    cmd = argparse.ArgumentParser("convert conll file to segment data.")
    cmd.add_argument('--input', default='../data/CTB5.1-train.gp.conll')
    cmd.add_argument('--output', default='../data/output_seg.txt')

    args = cmd.parse_args()
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
        chars, tags = [], []
        for line in codecs.open(args.input, "r", encoding='utf-8'):
            line = line.strip()
            if len(line) == 0:
                print(u'{0}\t{1}'.format(u' '.join(chars), u' '.join(tags)), file=output_file)
                chars, tags = [], []
            else:
                fields = line.split()
                word = fields[1]
                if len(word) == 1:
                    chars.append(word[0])
                    tags.append('S')
                else:
                    for i, ch in enumerate(word):
                        chars.append(ch)
                        if i == 0:
                            tags.append('B')
                        elif i == len(word) - 1:
                            tags.append('E')
                        else:
                            tags.append('I')
        if len(chars) > 0:
            print(u'{0}\t{1}'.format(u' '.join(chars), u' '.join(tags)), file=output_file)
    finally:
        output_file.close()


if __name__ == '__main__':
    main()
