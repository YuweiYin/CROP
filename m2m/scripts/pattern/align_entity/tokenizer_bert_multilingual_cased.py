import transformers
import sys


tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

for line in sys.stdin:
    output = " ".join(tokenizer.tokenize(line.strip())) + "\n"
    sys.stdout.write(output)
