# -*- coding: utf8 -*-
from __future__ import print_function


dbc = {
  "digit": {
    u"０", u"１", u"２", u"３", u"４", u"５", u"６", u"７", u"８", u"９",
    u"〇", u"一", u"二", u"三", u"四", u"五", u"六", u"七", u"八", u"九",
    u"零", u"壹", u"贰", u"叁", u"肆", u"伍", u"陆", u"柒", u"捌", u"玖"},
  "uppercase": {
    u"Ａ", u"Ｂ", u"Ｃ", u"Ｄ", u"Ｅ", u"Ｆ", u"Ｇ", u"Ｈ", u"Ｉ", u"Ｊ",
    u"Ｋ", u"Ｌ", u"Ｍ", u"Ｎ", u"Ｏ", u"Ｐ", u"Ｑ", u"Ｒ", u"Ｓ", u"Ｔ",
    u"Ｕ", u"Ｖ", u"Ｗ", u"Ｘ", u"Ｙ", u"Ｚ"},
  "lowercase": {
    u"ａ", u"ｂ", u"ｃ", u"ｄ", u"ｅ", u"ｆ", u"ｇ", u"ｈ", u"ｉ", u"ｊ",
    u"ｋ", u"ｌ", u"ｍ", u"ｎ", u"ｏ", u"ｐ", u"ｑ", u"ｒ", u"ｓ", u"ｔ",
    u"ｕ", u"ｖ", u"ｗ", u"ｘ", u"ｙ", u"ｚ"},
   "punct": {
     u"　", u"！", u"＂", u"＃", u"＄", u"％", u"＆", u"＇", u"（", u"）",
     u"＊", u"＋", u"，", u"－", u"．", u"／", u"：", u"；", u"＜", u"＝",
     u"＞", u"？", u"＠", u"［", u"＼", u"］", u"＾", u"＿", u"｀", u"｛",
     u"｜", u"｝", u"～"},
   "chinese-punct": {
     u"。", u"、", u"“",  u"”",  u"﹃", u"﹄", u"‘",  u"’",  u"﹁", u"﹂",
     u"…", u"【", u"】", u"《", u"》", u"〈", u"〉", u"·"},
}

sbc = {
  "digit": set([chr(ord('0') + i) for i in range(10)]),
  "uppercase": set([chr(ord('A') + i) for i in range(26)]),
  "lowercase": set([chr(ord('a') + i) for i in range(26)]),
  "punct": {
    ' ', '!', '"', '#', '$', '%', '&', '\'','(', ')', '*', '+', ',', '-',
    '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_',
    '`', '{', '|', '}', '~'}
}


# --- level 1 ---
CHAR_LETTER = 1
CHAR_DIGIT = 2
CHAR_PUNC = 3
CHAR_OTHER = -1

# --- level 2 ---
CHAR_LETTER_SBC = 11
CHAR_LETTER_DBC = 12

CHAR_DIGIT_SBC = 21
CHAR_DIGIT_DBC = 22

CHAR_PUNC_SBC = 31
CHAR_PUNC_DBC = 32

# --- level 3 ---
CHAR_LETTER_SBC_UPPERCASE = 111
CHAR_LETTER_SBC_LOWERCASE = 112
CHAR_LETTER_DBC_UPPERCASE = 121
CHAR_LETTER_DBC_LOWERCASE = 122

CHAR_DIGIT_DBC_CL1 = 221
CHAR_DIGIT_DBC_CL2 = 222
CHAR_DIGIT_DBC_CL3 = 223

CHAR_PUNC_DBC_NORMAL = 321
CHAR_PUNC_DBC_CHINESE = 322
CHAR_PUNC_DBC_EXT = 323


def chartype(ch):
  if ch in dbc['punct'] or ch in dbc['chinese-punct'] or ch in sbc['punct']:
    return CHAR_PUNC
  elif ch in dbc['uppercase'] or ch in sbc['uppercase']:
    return CHAR_LETTER
  elif ch in dbc['digit'] or ch in sbc['digit']:
    return CHAR_DIGIT
  else:
    return CHAR_OTHER

def gen(name, prefix, lex):
  print("static const int __chartype_%s_%s_utf8_size__ = %d;" % (prefix, name.replace("-", "_"), len(lex[name])))
  print("static const char* __chartype_%s_%s_utf8_buff__[] = {" % (prefix, name.replace("-", "_")))
  for i, k in enumerate(lex[name]):
    if k == "\"":
      k = '\\"'
    if k == "\\":
      k = "\\\\"
    print("\"%s\"," % k.encode("utf-8").__repr__()[1:-1], end='')
    if (i + 1) % 15 == 0:
      print()
  print("};")


if __name__ == "__main__":
  print("#ifndef __LTP_STRUTILS_CHARTYPES_TAB__\n"
        "#define __LTP_STRUTILS_CHARTYPES_TAB__\n"
        "#pragma warning(disable: 4309)")

  gen("chinese-punc", "dbc", dbc)
  gen("digit", "dbc", dbc)
  gen("punc", "dbc", dbc)
  gen("uppercase", "dbc", dbc)
  gen("punc-ext", "dbc", dbc)
  gen("lowercase", "dbc", dbc)
  gen("uppercase", "sbc", sbc)
  gen("lowercase", "sbc", sbc)
  gen("digit", "sbc", sbc)
  gen("punc", "sbc", sbc)

  print("#endif   // end for __LTP_STRUTILS_CHARTYPES_TAB__")
