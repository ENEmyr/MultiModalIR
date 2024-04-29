from torch import Tensor
from typing import List


class TextTransform:
    """Map characters to integers and vice versa"""

    def __init__(self):
        self.char_map = {}
        for i, char in enumerate(range(65, 91)):  # Uppercase Only
            self.char_map[chr(char)] = i
        self.char_map["'"] = 26
        self.char_map[" "] = 27
        self.index_map = {}
        for char, i in self.char_map.items():
            self.index_map[i] = char

    def text_to_int(self, text: str) -> List[int]:
        """Map text string to an integer sequence"""
        int_sequence = []
        for c in text:
            ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels: List[int | Tensor]) -> str:
        """Map integer sequence to text string"""
        string = []
        for i in labels:
            if type(i) == type(Tensor):
                i = i.item()
            if i == 28:  # blank char
                continue
            else:
                string.append(self.index_map[i])
        return "".join(string)


if __name__ == "__main__":
    tt = TextTransform()
    transformed_text = tt.text_to_int("down".upper())
    text = tt.int_to_text(transformed_text)
    print("trasformed text : ", transformed_text, "\ntext : ", text)
