# -*- coding: utf-8 -*-

import sys


class Word_to_Char():
    def __init__(self):
        self.word2char = {}
        self.char2word = {}
        self.max_utf_index = int('c280', 16) - 1
        self.char_unk = '無'

    # 単語一つをutf-8の一文字に対応させる
    # https://ja.wikipedia.org/wiki/UTF-8#%E3%83%90%E3%82%A4%E3%83%88%E9%A0%86%E3%83%9E%E3%83%BC%E3%82%AF%E3%81%AE%E4%BD%BF%E7%94%A8
    def add(self, word):
        if word in self.word2char:
            return
        while True:
            self.max_utf_index += 1
            # i byte は処理にとって何か重要な記号を含んでいる可能性があるので使用しない
            # 1 byte
            # if self.max_utf_index <= int('7f', 16):
            #     hex_str = format(self.max_utf_index, '02x')
            # elif self.max_utf_index == int('7f', 16) + 1:
            #     self.max_utf_index = int('c280', 16) - 1
            #     continue

            # 2 byte
            if self.max_utf_index <= int('dfbf', 16):
                hex_str = format(self.max_utf_index, '04x')
            elif self.max_utf_index == int('dfbf', 16) + 1:
                self.max_utf_index = int('e0a080', 16) - 1
                continue
            # 3 byte
            elif self.max_utf_index <= int('efbfbf', 16):
                hex_str = format(self.max_utf_index, '06x')
            elif self.max_utf_index == int('efbfbf', 16) + 1:
                self.max_utf_index = int('f0908080', 16) - 1
                continue
            # 4 byte
            elif self.max_utf_index <= int('f7bfbfbf', 16):
                hex_str = format(self.max_utf_index, '08x')
                # 使用可能な4 byte 文字検索の効率化
                if hex_str[2:8] == 'bfbfbf':
                    self.max_utf_index += (int('01000000', 16) - int('bfbfbf', 16) + int('808080', 16))
                    continue
                elif hex_str[4:8] == 'bfbf':
                    self.max_utf_index += (int('010000', 16) - int('00bfbf', 16) + int('008080', 16))
                    continue
            else:
                assert (False)
            hex_byte = bytes.fromhex(hex_str)
            try:
                utf_char = hex_byte.decode('utf-8')
                if utf_char == '\n' or utf_char == '\r' or utf_char == '\n\r' or \
                        utf_char == self.char_unk or utf_char.split() == '':
                    continue
                if len(utf_char) != 1:
                    continue
                break
            except UnicodeDecodeError:
                pass

        self.word2char[word] = utf_char
        return

    def get(self, word):
        if word not in self.word2char:
            return self.char_unk
        else:
            return self.word2char[word]


if __name__ == '__main__':
    w2c = Word_to_Char()
    for line in open(sys.argv[1], 'r'):
        line = line.rstrip()
        line_sp = line.split(' ')
        newline = ''
        for token in line_sp:
            w2c.add(token)
            newline += w2c.get(token)
        print(newline)
